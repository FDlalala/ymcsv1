import torch
import warnings
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

# ========== 1. Embeddings ==========
print("加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="./bge-small-zh-v1.5"
)

# ========== 2. 向量数据库 ==========
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_web"
)
doc_count = vectordb._collection.count()
print(f"向量库文档数量: {doc_count}")

# ========== 3. 本地 LLM ==========
model_path = "/data/models/Qwen2-7B-Instruct"
print("加载 LLM 模型...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 用 GenerationConfig 统一管理生成参数，消除警告
# 注意：不要在 pipeline() 里再传 max_new_tokens，否则两者冲突
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
    # max_new_tokens 已由 model.generation_config 统一管理，此处不再重复传入
)

llm = HuggingFacePipeline(pipeline=pipe)
print("LLM 加载完成！")

# ========== 4. Prompt 模板 ==========
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

# ========== 5. RAG Chain ==========
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # 从 3 增加到 4，减少因分块过碎导致的漏检
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ========== 6. 交互问答 ==========
print("\n" + "="*50)
print("RAG 问答系统已就绪，输入 quit 退出")
print("="*50)

while True:
    query = input("\n请输入问题: ").strip()
    if query.lower() in ["quit", "exit", "q"]:
        print("再见！")
        break
    if not query:
        continue

    print("\n检索中...")
    result = qa_chain.invoke({"query": query})

    answer = result["result"]
    sources = result["source_documents"]

    # ===== 后处理截断：防止模型在"不知道"后继续续写 =====
    REJECT_MARKER = "知识库中没有找到相关信息"
    if REJECT_MARKER in answer:
        answer = REJECT_MARKER

    print("\n" + "="*50)
    print("【回答】")
    print(answer)

    print("\n【参考来源】")
    seen = set()
    for i, doc in enumerate(sources, 1):
        src = doc.metadata.get("source", "未知")
        title = doc.metadata.get("title", "未知")
        if src not in seen:
            seen.add(src)
            print(f"  [{i}] {title}")
            print(f"      {src}")
            print(f"      摘要: {doc.page_content[:80].strip()}...")
    print("="*50)
