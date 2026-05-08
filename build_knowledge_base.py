"""
读取研报PDF → 切分 → 向量化 → 存入向量库
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

REPORTS_DIR = "./reports"
FAISS_DIR = "./faiss_db"

# 1. 读取所有PDF
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    print(f"请把研报PDF放到 {REPORTS_DIR} 文件夹下")
    exit()

pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith('.pdf')]
if len(pdf_files) == 0:
    print("未找到PDF文件，请先下载研报")
    exit()

print(f"找到 {len(pdf_files)} 份PDF文件")
all_docs = []
for pdf_file in pdf_files:
    file_path = os.path.join(REPORTS_DIR, pdf_file)
    try:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  ✓ {pdf_file}")
    except Exception as e:
        print(f"  ✗ {pdf_file} 读取失败: {e}")

print(f"\n共读取 {len(all_docs)} 页内容")

# 2. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = splitter.split_documents(all_docs)
print(f"切分成 {len(chunks)} 个文本片段")

# 3. 向量化 + 入库（FAISS，轻量无依赖问题）
print("正在加载Embedding模型并入库...")
embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectordb = FAISS.from_documents(documents=chunks, embedding=embedding)
vectordb.save_local(FAISS_DIR)
print(f"向量库已保存到 {FAISS_DIR}")
