"""
从PDF第一页提取证券机构名称，生成元数据
"""
import os
import json
import re

REPORTS_DIR = "./reports"
META_FILE = "./reports_meta.json"

# 常见证券机构关键词
BROKERS = [
    "国信证券", "中信证券", "华泰证券", "招商证券", "广发证券",
    "申万宏源", "海通证券", "中金公司", "银河证券", "国泰君安",
    "兴业证券", "东方证券", "光大证券", "平安证券", "长江证券",
    "天风证券", "东吴证券", "国金证券", "浙商证券", "华创证券",
    "中信建投", "中银证券", "开源证券", "民生证券", "德邦证券",
    "东海证券", "东兴证券", "方正证券", "财通证券", "华安证券",
    "国联证券", "山西证券", "东北证券", "东方财富", "万联证券",
    "国盛证券", "信达证券", "中泰证券", "西南证券", "国海证券",
    "华鑫证券", "中原证券", "太平洋证券", "国元证券", "华西证券",
    "南京证券", "长城证券", "东莞证券", "华福证券", "爱建证券",
    "财信证券", "高盛", "摩根士丹利", "瑞银", "摩根大通",
]

def extract_org_from_first_page(filepath):
    """读取PDF第一页，尝试提取证券机构名"""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            if not pdf.pages:
                return ""
            text = pdf.pages[0].extract_text() or ""
            # 从前往后找第一个匹配的券商名
            for broker in BROKERS:
                if broker in text[:2000]:
                    return broker
            # 尝试找"证券"关键词
            match = re.search(r"[一-鿿]{2,4}证券", text[:2000])
            if match:
                return match.group()
    except:
        pass
    return ""

def extract_title_from_first_page(filepath):
    """提取报告标题（通常是第一页最大字号的那行文字）"""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            if not pdf.pages:
                return ""
            text = pdf.pages[0].extract_text() or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # 跳过页眉、免责声明等
            skip_keywords = ["证券研究报告", "请阅读最后一页", "免责声明", "重要提示",
                           "本报告由", "投资评级说明", "股票投资评级", "行业投资评级"]
            # 找第二行或第三行（常见格式：第一行证券名，第二行报告标题）
            for line in lines[1:5]:
                if any(k in line for k in skip_keywords):
                    continue
                if len(line) > 10 and len(line) < 80:
                    return line.strip()
            # 没找到的话，取第一个有实际内容的行
            for line in lines[:5]:
                if len(line) > 5 and not any(k in line for k in skip_keywords):
                    return line.strip()
    except:
        pass
    return ""


print("扫描PDF并提取元数据...")
meta = {}
total = 0
for fname in os.listdir(REPORTS_DIR):
    if not fname.lower().endswith('.pdf'):
        continue
    total += 1
    fpath = os.path.join(REPORTS_DIR, fname)

    # 尝试从文件名解析（auto_update格式）
    match = re.match(r"(\d{8})_(\d{6})_(.+)\.pdf$", fname)
    if match:
        date, code, org = match.groups()
        meta[fname] = {"org": org, "title": fname, "stock": "", "date": date}
        continue

    # 从PDF内容提取
    org = extract_org_from_first_page(fpath)
    title = extract_title_from_first_page(fpath)
    meta[fname] = {"org": org, "title": title, "stock": "", "date": ""}

    flag = "✓" if org else " "
    short_title = title[:25] if title else ""
    print(f"  [{flag}] {fname[:30]:30s} | {org or '未知机构':12s} | {short_title}")

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

with_org = sum(1 for v in meta.values() if v["org"])
print(f"\n完成：共 {total} 份研报，识别出机构 {with_org} 份")
