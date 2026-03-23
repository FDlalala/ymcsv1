import warnings
warnings.filterwarnings("ignore")
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ========== 第一步：读取 JSON 案例库 ==========
CASE_JSON_PATH = "./cases.json"  # 改成你的 JSON 文件路径

print("读取案例库 JSON...")
with open(CASE_JSON_PATH, "r", encoding="utf-8") as f:
    case_data = json.load(f)

documents = []
skipped = []

for case_id, case_info in case_data.items():
    case_name = case_info.get("case_name", "")
    text_list = case_info.get("text", [])

    # 将 text 列表拼接成完整文本（每段之间用换行分隔）
    full_text = "\n".join(text_list).strip()

    if full_text:
        documents.append(Document(
            page_content=full_text,
            metadata={
                "case_id": case_id,
                "case_name": case_name,
                "source": CASE_JSON_PATH
            }
        ))
        print(f"  ✅ [{case_id}] {case_name[:40]:<40} ({len(full_text):>6} 字符)")
    else:
        skipped.append(case_id)
        print(f"  ⏭️  跳过空案例: {case_id}")

print(f"\n成功加载: {len(documents)} 个案例")
print(f"跳过/失败: {len(skipped)} 个案例")

# ========== 第二步：写入向量库（加速版）==========
print("\n加载 Embedding 模型...")

# 优化1：开启多线程并行编码，encode_kwargs 中设置 batch_size 和多线程
model_kwargs = {"device": "cuda"}  # 如有 GPU 改为 "cuda"
encode_kwargs = {
    "batch_size": 256,          # 每批编码数量，显存/内存足够可调大
    "normalize_embeddings": True,
    "show_progress_bar": True,
}
embeddings = HuggingFaceEmbeddings(
    model_name="./bge-small-zh-v1.5",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

print("清空旧知识库，重新构建...")
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_cases"
)
vectordb.delete_collection()

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_cases"
)

# 优化2：预先批量计算所有 embedding，再一次性写入 Chroma
# 这样避免 Chroma 内部逐条调用 embed，充分利用向量化批处理
print("\n预计算所有文档的 Embedding（批量加速）...")
texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]

# 优化3：加大写入批次，减少 Chroma 的 I/O 次数
batch_size = 500  # 原来是 100，调大可减少写入轮次
total = len(documents)

for i in range(0, total, batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_metas = metadatas[i:i+batch_size]
    vectordb.add_texts(texts=batch_texts, metadatas=batch_metas)
    done = min(i + batch_size, total)
    print(f"  写入进度: {done}/{total} ({done * 100 // total}%)")

final_count = vectordb._collection.count()
print(f"\n✅ 案例知识库构建完成！共 {final_count} 条记录")
