"""
eval_agent_worker.py —— 真正的 Agentic RAG 评估 Worker
与 eval_rag_worker.py 的核心区别：
  - 使用 create_react_agent + @tool，LLM 自主决定是否调用检索工具
  - RetrievalQA 是固定流程（每次必检索），这里是 LLM 驱动的工具调用
  - 记录 tool_call_count（工具调用次数）和 tool_queries（每次检索的 query）
  - D 类领域外问题理论上不会触发检索（LLM 判断无需检索直接拒绝）

用法（由 eval_agent_parallel.py 自动调用）：
  CUDA_VISIBLE_DEVICES=<gpu_id> python eval_agent_worker.py \\
      --shard_id 0 \\
      --questions_json '[{"category":"A_exact_grounding","idx":1,"question":"..."}]' \\
      --save_dir ./eval_results_agent \\
      --timestamp 20260319_090000
"""

import argparse
import json
import time
import os
import re
import torch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--shard_id",       type=int,   required=True)
parser.add_argument("--questions_json", type=str,   required=True)
parser.add_argument("--save_dir",       type=str,   default="./eval_results_agent")
parser.add_argument("--timestamp",      type=str,   required=True)
args = parser.parse_args()

shard_id  = args.shard_id
questions = json.loads(args.questions_json)
save_dir  = args.save_dir
timestamp = args.timestamp

os.makedirs(save_dir, exist_ok=True)

shard_json = os.path.join(save_dir, f"shard_{shard_id:02d}_{timestamp}.json")
shard_txt  = os.path.join(save_dir, f"shard_{shard_id:02d}_{timestamp}.txt")

print(f"[Worker {shard_id}] GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  "
      f"分配题数={len(questions)}  结果文件={shard_json}", flush=True)

# ============================================================
# 配置
# ============================================================
RETRIEVE_K        = 4
REFUSE_MARKER     = "知识库中没有找到相关信息"
RETRIEVAL_HIT_THR = 0.5
CATEGORY_EXPECTED = {
    "A_exact_grounding": True,
    "B_reasoning":       True,
    "C_boundary":        True,
    "D_out_of_domain":   False,
}
# Agent 最大迭代步数（防止无限循环）
MAX_ITERATIONS = 5

# ============================================================
# 加载模型
# ============================================================
print(f"[Worker {shard_id}] 加载 Embedding...", flush=True)
embeddings = HuggingFaceEmbeddings(model_name="./bge-small-zh-v1.5")

print(f"[Worker {shard_id}] 加载向量库...", flush=True)
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_web"
)
doc_count = vectordb._collection.count()
print(f"[Worker {shard_id}] 向量库文档数量: {doc_count}", flush=True)

print(f"[Worker {shard_id}] 加载 LLM...", flush=True)
model_path = "/data/models/Qwen2-7B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model      = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.float16, device_map="auto", trust_remote_code=True
)
gen_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
model.generation_config = gen_config

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=pipe)
print(f"[Worker {shard_id}] 模型加载完成！", flush=True)

# ============================================================
# 定义检索工具
# 工具的 docstring 是 LLM 决定是否调用的关键依据
# ============================================================
# 用闭包持有 vectordb 和 embeddings 引用，同时记录本次检索的文档
_last_retrieved_docs = []

@tool
def retrieve_context(query: str) -> str:
    """
    从深度学习知识库中检索与问题相关的内容。
    当需要回答关于深度学习、神经网络、机器学习、计算机视觉等领域的问题时，调用此工具。
    如果问题明显超出知识库范围（如询问最新新闻、其他领域知识），则不需要调用此工具。
    参数 query：用于检索的关键词或问题描述。
    """
    global _last_retrieved_docs
    docs = vectordb.similarity_search(query, k=RETRIEVE_K)
    _last_retrieved_docs = docs  # 保存供后续评估使用

    if not docs:
        return "知识库中未找到相关内容。"

    result_parts = []
    for i, doc in enumerate(docs, 1):
        title  = doc.metadata.get("title",  "未知标题")
        source = doc.metadata.get("source", "未知来源")
        result_parts.append(
            f"[片段{i}] 来源：{title}\n{doc.page_content[:400]}"
        )
    return "\n\n".join(result_parts)


# ============================================================
# 构建 Agent
# system_prompt 告诉 LLM 它的角色和行为规范
# ============================================================
SYSTEM_PROMPT = """你是一个知识库问答助手，知识库内容涵盖深度学习、神经网络、机器学习等领域。

你有一个工具 retrieve_context 可以从知识库中检索相关内容。

【行为规则】
1. 对于深度学习、神经网络、机器学习、计算机视觉等领域的问题，先调用 retrieve_context 工具检索相关内容，再基于检索结果回答。
2. 如果检索结果与问题相关，请用中文简洁回答，不要重复问题。
3. 如果检索结果与问题完全无关，或问题明显超出知识库范围（如询问最新新闻、其他领域），请直接输出：知识库中没有找到相关信息
4. 不要编造知识库中没有的内容。
"""

agent = create_react_agent(
    llm,
    tools=[retrieve_context],
    prompt=SYSTEM_PROMPT,
)

# ============================================================
# 辅助函数
# ============================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_retrieval_relevance(question: str, docs: list) -> tuple:
    """用 Embedding 余弦相似度衡量检索相关性"""
    if not docs:
        return 0.0, False, []
    q_emb  = np.array(embeddings.embed_query(question))
    scores = []
    for doc in docs:
        d_emb = np.array(embeddings.embed_query(doc.page_content))
        scores.append(cosine_similarity(q_emb, d_emb))
    best_score = max(scores)
    hit        = best_score >= RETRIEVAL_HIT_THR
    return round(best_score, 4), hit, [round(s, 4) for s in scores]


def classify_answer(answer: str, expected_to_answer: bool) -> tuple:
    refused = REFUSE_MARKER in answer
    if refused:
        answer_type = "refused"
        error_type  = "SHOULD_ANSWER" if expected_to_answer else "OK"
    else:
        answer_type = "answered"
        error_type  = "HALLUCINATION" if not expected_to_answer else "OK"
    return answer_type, error_type


def check_source_quality(sources: list) -> tuple:
    seen = set()
    has_title = False
    for doc in sources:
        src   = doc.metadata.get("source", "")
        title = doc.metadata.get("title",  "")
        if src:
            seen.add(src)
        if title and title != "未知标题":
            has_title = True
    unique_count = len(seen)
    src_error = "DUPLICATE_SOURCE" if unique_count <= 1 and len(sources) > 1 else "OK"
    return unique_count, has_title, src_error


def parse_agent_output(messages: list) -> tuple:
    """
    从 Agent 消息历史中解析：
    - final_answer: 最终回答文本
    - tool_call_count: 工具调用次数
    - tool_queries: 每次工具调用的 query 参数列表
    - retrieval_triggered: 是否触发了检索
    """
    final_answer    = ""
    tool_call_count = 0
    tool_queries    = []

    for msg in messages:
        # 统计工具调用（AIMessage 中包含 tool_calls）
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_count += 1
                    # 提取 query 参数
                    args_dict = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    q = args_dict.get("query", "")
                    if q:
                        tool_queries.append(q)
            # 最后一条 AIMessage 的 content 是最终回答
            if msg.content:
                final_answer = msg.content

    retrieval_triggered = tool_call_count > 0
    return final_answer, tool_call_count, tool_queries, retrieval_triggered


def truncate_hallucination(answer: str) -> str:
    """后处理：截断幻觉续写（对话历史混入）"""
    stop_patterns = [
        r'\nHuman\s*:',
        r'\n用户\s*:',
        r'\n问题\s*:',
        r'\n请根据以下',
        r'\n翻译[：:]',
        r'\nassistant\s*\n',
    ]
    for pat in stop_patterns:
        m = re.search(pat, answer)
        if m:
            answer = answer[:m.start()].strip()
    return answer


# ============================================================
# 初始化分片结果文件
# ============================================================
with open(shard_json, "w", encoding="utf-8") as f:
    f.write("[\n")
with open(shard_txt, "w", encoding="utf-8") as f:
    f.write(f"{'='*70}\n")
    f.write(f"[Agentic RAG] Worker {shard_id}  GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  "
            f"时间={timestamp}\n")
    f.write(f"{'='*70}\n\n")

# ============================================================
# 评估循环
# ============================================================
_first = True

for i, item in enumerate(questions):
    category           = item["category"]
    idx                = item["idx"]
    question           = item["question"]
    expected_to_answer = CATEGORY_EXPECTED.get(category, True)

    print(f"\n[Worker {shard_id}] [{i+1}/{len(questions)}] {category} Q{idx}: "
          f"{question[:50]}{'...' if len(question)>50 else ''}", flush=True)

    # 每题重置全局文档缓存
    _last_retrieved_docs = []

    t0 = time.time()
    try:
        # 调用 Agent（LLM 自主决定是否检索）
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": MAX_ITERATIONS * 2 + 1},
        )
        elapsed = time.time() - t0

        messages = result.get("messages", [])

        # 解析 Agent 输出
        raw_answer, tool_call_count, tool_queries, retrieval_triggered = parse_agent_output(messages)

        # 后处理
        if REFUSE_MARKER in raw_answer:
            raw_answer = REFUSE_MARKER
        raw_answer = truncate_hallucination(raw_answer)

        # 检索质量评估（基于最后一次检索到的文档）
        docs = _last_retrieved_docs
        retrieval_score, retrieval_hit, doc_scores = check_retrieval_relevance(question, docs)

        # 回答分类
        answer_type, error_type = classify_answer(raw_answer, expected_to_answer)

        # 检索触发但未命中
        if retrieval_triggered and not retrieval_hit and error_type == "OK":
            error_type = "RETRIEVAL_MISS"

        # 来源质量
        unique_src_count, has_title, src_error = check_source_quality(docs)
        if src_error != "OK" and error_type == "OK":
            error_type = src_error

        source_list = [{
            "title":   doc.metadata.get("title",  "未知标题"),
            "source":  doc.metadata.get("source", "未知来源"),
            "snippet": doc.page_content[:120].replace("\n", " ").strip()
        } for doc in docs]

        record = {
            "system":              "agentic_rag",
            "shard_id":            shard_id,
            "category":            category,
            "question_idx":        idx,
            "question":            question,
            "retrieval_triggered": retrieval_triggered,   # LLM 是否主动调用了工具
            "tool_call_count":     tool_call_count,       # 工具调用次数（可能多轮）
            "tool_queries":        tool_queries,          # 每次检索用的 query
            "retrieval_score":     retrieval_score,
            "retrieval_hit":       retrieval_hit,
            "doc_scores":          doc_scores,
            "answer_type":         answer_type,
            "error_type":          error_type,
            "unique_source_count": unique_src_count,
            "has_title":           has_title,
            "elapsed_sec":         round(elapsed, 2),
            "raw_answer":          raw_answer,
            "sources":             source_list,
        }

    except Exception as e:
        elapsed = time.time() - t0
        import traceback
        print(f"[Worker {shard_id}] ❌ 异常: {e}", flush=True)
        traceback.print_exc()
        record = {
            "system":              "agentic_rag",
            "shard_id":            shard_id,
            "category":            category,
            "question_idx":        idx,
            "question":            question,
            "retrieval_triggered": False,
            "tool_call_count":     0,
            "tool_queries":        [],
            "retrieval_score":     0.0,
            "retrieval_hit":       False,
            "doc_scores":          [],
            "answer_type":         "error",
            "error_type":          f"EXCEPTION: {str(e)[:100]}",
            "unique_source_count": 0,
            "has_title":           False,
            "elapsed_sec":         round(elapsed, 2),
            "raw_answer":          "",
            "sources":             [],
        }

    # 控制台输出
    icon = "✅" if record["error_type"] == "OK" else "❌"
    print(f"  {icon} 回答类型: {record['answer_type']}  错误类型: {record['error_type']}", flush=True)
    print(f"     检索触发: {record['retrieval_triggered']}  工具调用次数: {record['tool_call_count']}", flush=True)
    print(f"     检索 queries: {record['tool_queries']}", flush=True)
    print(f"     相似度: {record['retrieval_score']}  命中(≥{RETRIEVAL_HIT_THR}): {record['retrieval_hit']}", flush=True)
    print(f"     各文档分数: {record['doc_scores']}", flush=True)
    print(f"     来源数量: {record['unique_source_count']}  耗时: {record['elapsed_sec']}s", flush=True)
    print(f"\n  ── 完整回答 ──", flush=True)
    print(record['raw_answer'], flush=True)
    print(f"\n  ── 检索来源 ──", flush=True)
    for j, s in enumerate(record['sources'], 1):
        print(f"  [{j}] 标题: {s['title']}", flush=True)
        print(f"       来源: {s['source']}", flush=True)
        print(f"       摘要: {s['snippet']}", flush=True)
    print("-" * 60, flush=True)

    # 实时写入 JSON
    with open(shard_json, "a", encoding="utf-8") as f:
        if not _first:
            f.write(",\n")
        json.dump(record, f, ensure_ascii=False, indent=2)
        _first = False

    # 实时写入 TXT
    with open(shard_txt, "a", encoding="utf-8") as f:
        f.write(f"\n[{category}] Q{idx}: {question}\n")
        f.write(f"  检索触发: {record['retrieval_triggered']}  工具调用次数: {record['tool_call_count']}\n")
        f.write(f"  检索 queries: {record['tool_queries']}\n")
        f.write(f"  相似度: {record['retrieval_score']}  命中(≥{RETRIEVAL_HIT_THR}): {record['retrieval_hit']}\n")
        f.write(f"  各文档分数: {record['doc_scores']}\n")
        f.write(f"  回答类型: {record['answer_type']}  错误类型: {record['error_type']}\n")
        f.write(f"  来源数量: {record['unique_source_count']}  耗时: {record['elapsed_sec']}s\n")
        f.write(f"  完整回答:\n")
        for line in record['raw_answer'].splitlines():
            f.write(f"    {line}\n")
        f.write(f"  来源:\n")
        for s in record['sources']:
            f.write(f"    - [{s['title']}] {s['source']}\n")
            f.write(f"      摘要: {s['snippet']}\n")
        f.write("-" * 70 + "\n")

# 关闭 JSON 数组
with open(shard_json, "a", encoding="utf-8") as f:
    f.write("\n]\n")

print(f"\n[Worker {shard_id}] 完成！结果已写入 {shard_json}", flush=True)
