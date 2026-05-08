"""
自动搜索研报 → 下载 → 去重 → 追加到向量库
"""
import os
import json
import time
import requests
import hashlib
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"
REPORTS_DIR = "./reports"
RECORD_FILE = "./download_record.json"

# ─── 要跟踪的锂电/新能源股票 ───
STOCKS = [
    {"code": "300750", "name": "宁德时代"},
    {"code": "002594", "name": "比亚迪"},
    {"code": "300014", "name": "亿纬锂能"},
    {"code": "002074", "name": "国轩高科"},
    {"code": "300450", "name": "先导智能"},
    {"code": "300568", "name": "星源材质"},
    {"code": "002709", "name": "天赐材料"},
    {"code": "300073", "name": "当升科技"},
    {"code": "603659", "name": "璞泰来"},
    {"code": "300769", "name": "德方纳米"},
]

# ─── 记录管理 ───
def load_record():
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloaded_pdfs": [], "last_run": None, "total": 0}

def save_record(record):
    record["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

# ─── 步骤1：搜索新研报 ───
def fetch_new_reports():
    import akshare as ak
    record = load_record()
    downloaded_urls = set(record.get("downloaded_pdfs", []))
    new_reports = []

    for stock in STOCKS:
        print(f"  搜索 {stock['name']}({stock['code']})...", end=" ")
        try:
            df = ak.stock_research_report_em(symbol=stock["code"])
            count = 0
            for _, row in df.iterrows():
                pdf_url = str(row.iloc[-1])
                if pdf_url not in downloaded_urls:
                    new_reports.append({
                        "stock_name": stock["name"],
                        "stock_code": stock["code"],
                        "title": str(row.iloc[2]),
                        "org": str(row.iloc[4]),
                        "date": str(row.iloc[-3]),
                        "pdf_url": pdf_url,
                    })
                    count += 1
            print(f"新增 {count} 篇")
            time.sleep(1)
        except Exception as e:
            print(f"失败: {e}")

    return new_reports

# ─── 步骤2：下载PDF ───
def download_pdfs(reports):
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    downloaded = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for rpt in reports:
        filename = f"{rpt['date']}_{rpt['stock_code']}_{rpt['org']}.pdf"
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        filepath = os.path.join(REPORTS_DIR, filename)

        if os.path.exists(filepath):
            downloaded.append(rpt["pdf_url"])
            continue

        print(f"  下载: {rpt['org']} - {rpt['title'][:20]}...", end=" ")
        try:
            resp = requests.get(rpt["pdf_url"], headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                downloaded.append(rpt["pdf_url"])
                print("✓")
            else:
                print(f"HTTP {resp.status_code}")
        except Exception as e:
            print(f"异常: {e}")

    return downloaded

# ─── 步骤3：新入库 ───
def add_new_pdfs(downloaded_urls):
    if not downloaded_urls:
        print("没有新PDF需要入库")
        return

    # 找出新下载对应的文件
    new_files = []
    for fname in os.listdir(REPORTS_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        fpath = os.path.join(REPORTS_DIR, fname)
        new_files.append(fpath)

    if not new_files:
        return

    print(f"  解析 {len(new_files)} 份PDF...")
    all_docs = []
    for fpath in new_files:
        try:
            loader = PDFPlumberLoader(fpath)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ✗ {os.path.basename(fpath)} 解析失败: {e}")

    if not all_docs:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  切分成 {len(chunks)} 个片段")

    print("  追加到向量库...")
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    vectordb.add_documents(chunks)
    vectordb.persist()
    print(f"  ✓ 入库完成，新增 {len(downloaded_urls)} 篇")


# ─── 主流程 ───
def run():
    print("=" * 50)
    print(f"  研报自动更新 ｜ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    print("\n▶ 步骤1：搜索新研报")
    new_reports = fetch_new_reports()
    print(f"\n  共发现 {len(new_reports)} 篇新研报")

    if not new_reports:
        print("  没有新研报，跳过")
        return

    print("\n▶ 步骤2：下载PDF")
    downloaded = download_pdfs(new_reports)

    if downloaded:
        record = load_record()
        record["downloaded_pdfs"].extend(downloaded)
        record["total"] = len(record["downloaded_pdfs"])
        save_record(record)

    print(f"\n▶ 步骤3：追加到向量库")
    add_new_pdfs(downloaded)

    print(f"\n{'=' * 50}")
    print(f"  完成！本次新增 {len(downloaded)} 篇研报")
    print(f"  累计已收录 {load_record()['total']} 篇")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run()
