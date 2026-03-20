"""
eval_rag.py —— RAG 系统评估脚本
评估对象：rag_local.py（baseline RetrievalQA）
评估问题：eval_questions.py 中的四类问题
评估维度：
  1. retrieval_triggered  : 是否触发了检索（baseline 始终触发，agent 可能不触发）
  2. retrieval_hit        : 检索到的文档是否与问题相关（人工/关键词判断）
  3. answer_type          : 回答类型 —— "answered" / "refused" / "hallucination_risk"
  4. source_quality       : 来源是否合理（去重后来源数量、是否有标题）
  5. error_type           : 失败类型标注（见下方枚举）
  6. raw_answer           : 原始回答文本
  7. sources              : 参考来源列表

错误类型枚举：
  OK                  : 正常
  RETRIEVAL_MISS      : 检索错了（文档与问题无关）
  GENERATION_ERROR    : 检索对了但生成总结错了
  HALLUCINATION       : 书里没答案，模型还是乱说（应拒绝但未拒绝）
  SHOULD_ANSWER       : 书里有答案，模型却拒绝回答
  DUPLICATE_SOURCE    : 来源重复或不够代表性
  NO_RETRIEVAL        : 未触发检索（仅 agent 模式可能出现）
"""

import json
import time
import datetime
import os
import torch
import warnings
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

from eval_questions import QUESTIONS

# ============================================================
# 配置
# ============================================================
RETRIEVE_K      = 4          # 检索 top-k
SAVE_DIR        = "./eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 拒绝回答的标志词
REFUSE_MARKER   = "知识库中没有找到相关信息"

# 各类别的"预期行为"：True=应该回答，False=应该拒绝
CATEGORY_EXPECTED = {
    "A_exact_grounding": True,   # 书里明确有答案 → 应该回答
    "B_reasoning":       True,   # 书里相关 → 应该回答
    "C_boundary":        True,   # 书里相关但边界 → 应该回答（部分可能拒绝）
    "D_out_of_domain":   False,  # 书里没有 → 应该拒绝
}

# ============================================================
# 1. 加载模型（只加载一次）
# ============================================================
print("=" * 60)
print("加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="./bge-small-zh-v1.5")

print("加载向量数据库...")
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_web"
)
print(f"向量库文档数量: {vectordb._collection.count()}")

print("加载 LLM 模型...")
model_path = "/data/models/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
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
print("LLM 加载完成！\n")

# ============================================================
# 2. 构建 RAG Chain（baseline）
# ============================================================
prompt_template = """你是一个知识库问答助手。请根据下方【检索内容】回答用户问题。

【检索内容】
{context}

【问题】
{question}

【回答规则】
- 规则1：【检索内容】中可能使用不同的表述方式，请结合语义理解进行回答。
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
# 3. 辅助函数
# ============================================================

def check_retrieval_relevance(question: str, docs: list) -> bool:
    """
    简单关键词判断：检索到的文档是否与问题相关。
    策略：取问题中的关键词，看是否出现在任意一篇文档内容中。
    """
    # 提取问题中长度>=2的中文词（简单分词）
    import re
    # 去掉标点，取所有长度>=2的连续汉字片段
    tokens = re.findall(r'[\u4e00-\u9fff]{2,}', question)
    if not tokens:
        return True  # 无法判断，默认相关
    combined_content = " ".join(doc.page_content for doc in docs)
    # 只要有一个关键词命中，就认为相关
    for tok in tokens:
        if tok in combined_content:
            return True
    return False


def classify_answer(answer: str, expected_to_answer: bool) -> tuple:
    """
    返回 (answer_type, error_type)
    answer_type: "answered" / "refused"
    error_type:  "OK" / "HALLUCINATION" / "SHOULD_ANSWER"
    """
    refused = REFUSE_MARKER in answer

    if refused:
        answer_type = "refused"
        if expected_to_answer:
            error_type = "SHOULD_ANSWER"
        else:
            error_type = "OK"
    else:
        answer_type = "answered"
        if not expected_to_answer:
            # 书里没有，但模型回答了 → 幻觉风险
            error_type = "HALLUCINATION"
        else:
            error_type = "OK"

    return answer_type, error_type


def check_source_quality(sources: list) -> tuple:
    """
    返回 (unique_count, has_title, source_error)
    """
    seen_sources = set()
    has_title = False
    for doc in sources:
        src = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        if src:
            seen_sources.add(src)
        if title and title != "未知标题":
            has_title = True

    unique_count = len(seen_sources)
    source_error = "DUPLICATE_SOURCE" if unique_count <= 1 and len(sources) > 1 else "OK"
    return unique_count, has_title, source_error


# ============================================================
# 4. 实时写入文件初始化
# ============================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_path   = os.path.join(SAVE_DIR, f"eval_detail_{timestamp}.json")
report_path = os.path.join(SAVE_DIR, f"eval_report_{timestamp}.txt")

# 预先写入 JSON 开头（数组起始括号）
with open(json_path, "w", encoding="utf-8") as f:
    f.write("[\n")

# 预先写入 TXT 报告头
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(f"RAG 评估报告  生成时间: {timestamp}\n")
    f.write("=" * 70 + "\n\n")
    f.write("（本文件实时追加写入，每道题完成后立即更新）\n\n")

print(f"实时结果文件:\n  JSON : {json_path}\n  报告 : {report_path}\n")

# ============================================================
# 5. 主评估循环
# ============================================================
all_results = []
category_stats = {}
_json_first_record = True   # 用于控制 JSON 数组逗号

total_q = sum(len(qs) for qs in QUESTIONS.values())
done = 0

print("=" * 60)
print(f"开始评估，共 {total_q} 道题")
print("=" * 60)

for category, questions in QUESTIONS.items():
    expected_to_answer = CATEGORY_EXPECTED.get(category, True)
    cat_results = []

    print(f"\n{'='*60}")
    print(f"【类别】{category}  (预期行为: {'应回答' if expected_to_answer else '应拒绝'})")
    print(f"{'='*60}")

    for idx, question in enumerate(questions, 1):
        done += 1
        print(f"\n[{done}/{total_q}] Q: {question[:60]}{'...' if len(question)>60 else ''}")

        t0 = time.time()
        try:
            result = qa_chain.invoke({"query": question})
            elapsed = time.time() - t0

            raw_answer = result["result"]
            sources    = result["source_documents"]

            # 后处理：截断拒绝标记后的续写
            if REFUSE_MARKER in raw_answer:
                raw_answer = REFUSE_MARKER

            # 检索始终触发（baseline）
            retrieval_triggered = True

            # 检索相关性
            retrieval_hit = check_retrieval_relevance(question, sources)

            # 回答分类
            answer_type, error_type = classify_answer(raw_answer, expected_to_answer)

            # 如果检索没命中，覆盖 error_type
            if not retrieval_hit and error_type == "OK":
                error_type = "RETRIEVAL_MISS"

            # 来源质量
            unique_src_count, has_title, src_error = check_source_quality(sources)
            if src_error != "OK" and error_type == "OK":
                error_type = src_error

            # 整理来源列表
            source_list = []
            for doc in sources:
                source_list.append({
                    "title":   doc.metadata.get("title",  "未知标题"),
                    "source":  doc.metadata.get("source", "未知来源"),
                    "snippet": doc.page_content[:100].replace("\n", " ").strip()
                })

            record = {
                "category":            category,
                "question_idx":        idx,
                "question":            question,
                "retrieval_triggered": retrieval_triggered,
                "retrieval_hit":       retrieval_hit,
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
            record = {
                "category":            category,
                "question_idx":        idx,
                "question":            question,
                "retrieval_triggered": False,
                "retrieval_hit":       False,
                "answer_type":         "error",
                "error_type":          f"EXCEPTION: {str(e)[:100]}",
                "unique_source_count": 0,
                "has_title":           False,
                "elapsed_sec":         round(elapsed, 2),
                "raw_answer":          "",
                "sources":             [],
            }

        # ── 控制台完整输出（不省略）──────────────────────────────
        icon = "✅" if record["error_type"] == "OK" else "❌"
        print(f"  {icon} 回答类型: {record['answer_type']}  |  错误类型: {record['error_type']}")
        print(f"     检索触发: {record['retrieval_triggered']}  |  检索命中: {record['retrieval_hit']}")
        print(f"     来源数量: {record['unique_source_count']}  |  耗时: {record['elapsed_sec']}s")
        print("\n  ── 完整回答 ──")
        print(record['raw_answer'])
        print("\n  ── 检索来源 ──")
        for i, s in enumerate(record['sources'], 1):
            print(f"  [{i}] 标题: {s['title']}")
            print(f"       来源: {s['source']}")
            print(f"       摘要: {s['snippet']}")
        print("-" * 60)

        cat_results.append(record)
        all_results.append(record)

        # ── 实时写入 JSON（追加模式）────────────────────────────────
        with open(json_path, "a", encoding="utf-8") as f:
            if not _json_first_record:
                f.write(",\n")
            json.dump(record, f, ensure_ascii=False, indent=2)
            _json_first_record = False

        # ── 实时写入 TXT 报告（追加模式）────────────────────────────
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{record['category']}] Q{record['question_idx']}: {record['question']}\n")
            f.write(f"  检索触发: {record['retrieval_triggered']}  命中: {record['retrieval_hit']}\n")
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

    # 类别统计
    ok_count   = sum(1 for r in cat_results if r["error_type"] == "OK")
    err_counts = {}
    for r in cat_results:
        et = r["error_type"]
        err_counts[et] = err_counts.get(et, 0) + 1

    category_stats[category] = {
        "total":      len(cat_results),
        "ok":         ok_count,
        "ok_rate":    round(ok_count / len(cat_results), 3) if cat_results else 0,
        "error_dist": err_counts,
    }

    print(f"\n  ── 类别小结: OK={ok_count}/{len(cat_results)}  错误分布={err_counts}")

    # 实时写入类别小结到 TXT
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"类别小结 [{category}]\n")
        f.write(f"  题数: {len(cat_results)}  OK: {ok_count}  OK率: {ok_count/len(cat_results)*100:.1f}%\n")
        f.write(f"  错误分布: {err_counts}\n")
        f.write(f"{'='*70}\n")

# ============================================================
# 6. 收尾：关闭 JSON 数组 + 写入总结
# ============================================================
# 关闭 JSON 数组
with open(json_path, "a", encoding="utf-8") as f:
    f.write("\n]\n")

# 写入 TXT 总结
total_ok = sum(1 for r in all_results if r["error_type"] == "OK")
global_err = {}
for r in all_results:
    et = r["error_type"]
    global_err[et] = global_err.get(et, 0) + 1

with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 70 + "\n")
    f.write("总体统计\n")
    f.write("=" * 70 + "\n")
    f.write(f"总题数: {len(all_results)}   总体 OK 率: {total_ok}/{len(all_results)} "
            f"({100*total_ok/len(all_results):.1f}%)\n\n")
    f.write("全局错误类型分布:\n")
    for et, cnt in sorted(global_err.items(), key=lambda x: -x[1]):
        f.write(f"  {et:<25} : {cnt}\n")
    f.write("\n各类别统计:\n")
    for cat, stat in category_stats.items():
        f.write(f"  {cat:<25} OK率: {stat['ok_rate']*100:.1f}%  错误: {stat['error_dist']}\n")

print(f"\n详细结果已保存: {json_path}")
print(f"可读报告已保存: {report_path}")

# ============================================================
# 7. 控制台总结
# ============================================================
print("\n" + "=" * 60)
print("评估完成！总体统计：")
print("=" * 60)
for cat, stat in category_stats.items():
    print(f"  {cat:<25} OK率: {stat['ok_rate']*100:.1f}%  错误: {stat['error_dist']}")
print(f"\n总体 OK 率: {total_ok}/{len(all_results)} ({100*total_ok/len(all_results):.1f}%)")
print(f"\n结果文件:\n  {json_path}\n  {report_path}")
