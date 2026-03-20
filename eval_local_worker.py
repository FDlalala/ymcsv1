"""
eval_local_worker.py —— 针对 rag_local.py（RetrievalQA 链）的单进程评估 Worker
用法（由 eval_local_parallel.py 自动调用）：
  CUDA_VISIBLE_DEVICES=<gpu_id> python eval_local_worker.py \
      --shard_id 0 \
      --questions_json '[{"category":"A_exact_grounding","idx":1,"question":"..."}]' \
      --save_dir ./eval_results_local \
      --timestamp 20260319_080000
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
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--shard_id",       type=int, required=True)
parser.add_argument("--questions_json", type=str, required=True)
parser.add_argument("--save_dir",       type=str, default="./eval_results_local")
parser.add_argument("--timestamp",      type=str, required=True)
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
RETRIEVE_K         = 4
REFUSE_MARKER      = "知识库中没有找到相关信息"
RETRIEVAL_HIT_THR  = 0.5   # 余弦相似度阈值：≥ 此值视为命中
CATEGORY_EXPECTED = {
    "A_exact_grounding": True,
    "B_reasoning":       True,
    "C_boundary":        True,
    "D_out_of_domain":   False,
}

# ============================================================
# 加载模型（与 rag_local.py 完全一致）
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
# 构建 RAG Chain（与 rag_local.py 完全一致）
# ============================================================
prompt_template = """你是一个知识库问答助手。请根据下方【检索内容】回答用户问题。

【检索内容】
{context}

【问题】
{question}

【回答规则】
- 规则1：【检索内容】中可能使用不同的表述方式（例如"计算机视觉"和"机器视觉"指同一领域），请结合语义理解进行回答，不要因为措辞不完全一致就拒绝回答。
- 规则2：如果【检索内容】中包含与问题相关的信息，请直接用中文简洁回答，不要重复问题。
- 规则3：只有当【检索内容】与问题完全无关时，才输出：知识库中没有找到相关信息

【回答】"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVE_K}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ============================================================
# 辅助函数
# ============================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_retrieval_relevance(question: str, docs: list) -> tuple:
    """
    用 Embedding 余弦相似度衡量检索相关性。
    返回:
        retrieval_score  : float，问题与最相关文档的余弦相似度（0~1）
        retrieval_hit    : bool，score >= RETRIEVAL_HIT_THR
        doc_scores       : list[float]，每个文档的相似度分数
    """
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
    """判断回答类型和错误类型"""
    refused = REFUSE_MARKER in answer
    if refused:
        answer_type = "refused"
        error_type  = "SHOULD_ANSWER" if expected_to_answer else "OK"
    else:
        answer_type = "answered"
        error_type  = "HALLUCINATION" if not expected_to_answer else "OK"
    return answer_type, error_type


def check_source_quality(sources: list) -> tuple:
    """检查来源质量：是否全部重复"""
    seen = set()
    for doc in sources:
        src = doc.metadata.get("source", "")
        if src:
            seen.add(src)
    unique_count = len(seen)
    src_error = "DUPLICATE_SOURCE" if unique_count <= 1 and len(sources) > 1 else "OK"
    return unique_count, src_error


def truncate_hallucination(answer: str) -> str:
    """
    后处理：截断幻觉续写。
    rag_local.py 已对 REFUSE_MARKER 做了截断，这里额外处理
    模型在正常回答后继续生成对话历史的情况。
    """
    # 截断常见的续写标记
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
    f.write(f"[rag_local] Worker {shard_id}  GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  "
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

    t0 = time.time()
    try:
        result     = qa_chain.invoke({"query": question})
        elapsed    = time.time() - t0
        raw_answer = result["result"]
        sources    = result["source_documents"]

        # 后处理1：截断拒绝标记后的续写（与 rag_local.py 一致）
        if REFUSE_MARKER in raw_answer:
            raw_answer = REFUSE_MARKER

        # 后处理2：截断幻觉续写（对话历史混入）
        raw_answer = truncate_hallucination(raw_answer)

        retrieval_triggered  = True   # RetrievalQA 每次都触发检索
        retrieval_score, retrieval_hit, doc_scores = check_retrieval_relevance(question, sources)
        answer_type, error_type = classify_answer(raw_answer, expected_to_answer)

        if not retrieval_hit and error_type == "OK":
            error_type = "RETRIEVAL_MISS"

        unique_src_count, src_error = check_source_quality(sources)
        if src_error != "OK" and error_type == "OK":
            error_type = src_error

        source_list = [{
            "title":   doc.metadata.get("title",  "未知标题"),
            "source":  doc.metadata.get("source", "未知来源"),
            "snippet": doc.page_content[:120].replace("\n", " ").strip()
        } for doc in sources]

        record = {
            "system":              "rag_local",
            "shard_id":            shard_id,
            "category":            category,
            "question_idx":        idx,
            "question":            question,
            "retrieval_triggered": retrieval_triggered,
            "retrieval_score":     retrieval_score,   # 最高余弦相似度
            "retrieval_hit":       retrieval_hit,     # score >= 阈值
            "doc_scores":          doc_scores,        # 每个文档的相似度
            "answer_type":         answer_type,
            "error_type":          error_type,
            "unique_source_count": unique_src_count,
            "elapsed_sec":         round(elapsed, 2),
            "raw_answer":          raw_answer,
            "sources":             source_list,
        }

    except Exception as e:
        elapsed = time.time() - t0
        record = {
            "system":              "rag_local",
            "shard_id":            shard_id,
            "category":            category,
            "question_idx":        idx,
            "question":            question,
            "retrieval_triggered": False,
            "retrieval_score":     0.0,
            "retrieval_hit":       False,
            "doc_scores":          [],
            "answer_type":         "error",
            "error_type":          f"EXCEPTION: {str(e)[:100]}",
            "unique_source_count": 0,
            "elapsed_sec":         round(elapsed, 2),
            "raw_answer":          "",
            "sources":             [],
        }

    # 控制台完整输出
    icon = "✅" if record["error_type"] == "OK" else "❌"
    print(f"  {icon} 回答类型: {record['answer_type']}  错误类型: {record['error_type']}", flush=True)
    print(f"     检索触发: {record['retrieval_triggered']}  "
          f"相似度: {record['retrieval_score']}  命中(≥{RETRIEVAL_HIT_THR}): {record['retrieval_hit']}", flush=True)
    print(f"     各文档分数: {record['doc_scores']}", flush=True)
    print(f"     来源数量: {record['unique_source_count']}  耗时: {record['elapsed_sec']}s", flush=True)
    print(f"\n  ── 完整回答 ──", flush=True)
    print(record["raw_answer"], flush=True)
    print(f"\n  ── 检索来源 ──", flush=True)
    for j, s in enumerate(record["sources"], 1):
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
        f.write(f"  检索触发: {record['retrieval_triggered']}  "
                f"相似度: {record['retrieval_score']}  命中(≥{RETRIEVAL_HIT_THR}): {record['retrieval_hit']}\n")
        f.write(f"  各文档分数: {record['doc_scores']}\n")
        f.write(f"  回答类型: {record['answer_type']}  错误类型: {record['error_type']}\n")
        f.write(f"  来源数量: {record['unique_source_count']}  耗时: {record['elapsed_sec']}s\n")
        f.write(f"  完整回答:\n")
        for line in record["raw_answer"].splitlines():
            f.write(f"    {line}\n")
        f.write(f"  来源:\n")
        for s in record["sources"]:
            f.write(f"    - [{s['title']}] {s['source']}\n")
            f.write(f"      摘要: {s['snippet']}\n")
        f.write("-" * 70 + "\n")

# 关闭 JSON 数组
with open(shard_json, "a", encoding="utf-8") as f:
    f.write("\n]\n")

print(f"\n[Worker {shard_id}] 完成！结果已写入 {shard_json}", flush=True)
