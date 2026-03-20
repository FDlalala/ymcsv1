import warnings
warnings.filterwarnings("ignore")
import requests
import time
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# ========== 第一步：从目录页获取所有链接 ==========
BASE_URL = "https://zh.d2l.ai"
TOC_URL = "https://zh.d2l.ai/index.html"

def get_all_chapter_urls():
    """从目录页抓取所有章节链接"""
    print("获取全书目录...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(TOC_URL, headers=headers, timeout=15)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    seen = set()

    # 找所有章节链接
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # 只要 .html 页面，排除外部链接
        if href.endswith(".html") and not href.startswith("http"):
            full_url = BASE_URL + "/" + href.lstrip("./")
            if full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)

    print(f"找到 {len(urls)} 个页面链接")
    return urls

def fetch_page(url):
    """爬取单个页面正文"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 提取正文区域
        main = (
            soup.find("div", {"class": "bd-article"}) or
            soup.find("div", {"role": "main"}) or
            soup.find("article") or
            soup.find("main")
        )

        if not main:
            return None, None

        # 清理无用标签
        for tag in main.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = main.get_text(separator="\n", strip=True)
        title = soup.title.string.strip() if soup.title else url

        return text, title

    except Exception as e:
        return None, None

# ========== 第二步：爬取所有页面 ==========
urls = get_all_chapter_urls()

print(f"\n开始爬取 {len(urls)} 个页面...\n")
documents = []
failed = []

for i, url in enumerate(urls, 1):
    text, title = fetch_page(url)

    if text and len(text) > 200:
        documents.append(Document(
            page_content=text,
            metadata={
                "source": url,
                "title": title,
                "language": "zh"
            }
        ))
        print(f"[{i:3d}/{len(urls)}] ✅ {title[:45]:<45} ({len(text):>6} 字符)")
    else:
        failed.append(url)
        print(f"[{i:3d}/{len(urls)}] ⏭️  跳过: {url}")

    # 礼貌爬取，避免被封
    time.sleep(0.3)

print(f"\n成功爬取: {len(documents)} 页")
print(f"跳过/失败: {len(failed)} 页")

# ========== 第三步：文本分割 ==========
print("\n分割文本...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"共分割为 {len(chunks)} 个片段")

# ========== 第四步：写入向量库 ==========
print("\n加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="./bge-small-zh-v1.5")

print("清空旧知识库，重新构建...")
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_web"
)
vectordb.delete_collection()

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_web"
)

# 批量写入
batch_size = 100
total = len(chunks)
for i in range(0, total, batch_size):
    batch = chunks[i:i+batch_size]
    vectordb.add_documents(batch)
    print(f"  写入进度: {min(i+batch_size, total)}/{total} ({min(i+batch_size, total)*100//total}%)")

final_count = vectordb._collection.count()
print(f"\n✅ 知识库构建完成！共 {final_count} 条记录")

# ========== 第五步：保存已爬取的URL列表 ==========
with open("crawled_urls.txt", "w") as f:
    for doc in documents:
        f.write(doc.metadata["source"] + "\n")
print(f"已保存 URL 列表到 crawled_urls.txt")
