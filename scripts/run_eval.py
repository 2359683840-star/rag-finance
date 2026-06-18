"""
CLI — 运行来源锚定评测
不依赖LLM主观打分，所有指标可追溯到真实研报来源。

用法:
  python scripts/run_eval.py --build-benchmark     # 构建Benchmark
  python scripts/run_eval.py --full                 # 完整评测流程
  python scripts/run_eval.py --full --label "DeepSeek-v3"
"""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
import json
import time
from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from evaluation.benchmark import BenchmarkBuilder
from evaluation.judge import LLMJudge
from evaluation.reporter import EvalReporter

# 复用RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_DIR, EMBEDDING_MODEL


def build_benchmark():
    print("=" * 60)
    print("  构建来源锚定评测Benchmark")
    print("  类别: 检索评测 · 忠实度评测 · 事实准确性评测")
    print("=" * 60)
    bb = BenchmarkBuilder()
    bb.build(n_per_category=15)
    stats = bb.stats()
    print(f"\nBenchmark统计: {stats}")
    bb.export()
    return bb.test_cases


def generate_rag_responses(test_cases):
    """用RAG流水线为评测生成模型回答"""
    print("\n" + "=" * 60)
    print("  RAG生成模型回答（用于评测）")
    print("=" * 60)

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # 为检索评测：生成检索结果
    retrieved_docs_map = {}
    # 为忠实度/事实评测：生成模型回答
    model_answers = {}

    for i, case in enumerate(test_cases):
        cat = case.get("category", "")
        case_id = case.get("id", str(i))
        short = case.get('query') or case.get('claim') or case.get('question', '')
        print(f"  [{i+1}/{len(test_cases)}] {cat}: {str(short)[:60]}")

        if cat == "retrieval":
            query = case.get("query", "")
            docs = db.similarity_search(query, k=10)
            retrieved_docs_map[query] = [
                {
                    "content": d.page_content.strip()[:500],
                    "org": d.metadata.get("org", ""),
                    "title": d.metadata.get("title", ""),
                }
                for d in docs
            ]

        elif cat == "faithfulness":
            claim = case.get("claim", "")
            source = case.get("source_content", "")
            prompt = f"""请判断以下声明是否被原文支持。仅输出 "supported" 或 "not_supported"。

原文：{source[:1000]}

声明：{claim}

判断（只输出 supported 或 not_supported）："""
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                model_answers[case_id] = resp.choices[0].message.content.strip()
            except Exception as e:
                model_answers[case_id] = f"[错误: {e}]"

        elif cat == "factual":
            question = case.get("question", "")
            # 检索相关文档
            docs = db.similarity_search(question, k=5)
            context = "\n\n".join(
                f"[{d.metadata.get('org', '')}] {d.page_content[:500]}"
                for d in docs
            )
            prompt = f"""基于以下资料回答问题。如果资料不足，请说明。

资料：{context[:2500]}

问题：{question}

回答："""
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                model_answers[case_id] = resp.choices[0].message.content
            except Exception as e:
                model_answers[case_id] = f"[错误: {e}]"

        time.sleep(0.3)

    return retrieved_docs_map, model_answers


def run_evaluation(test_cases, retrieved_docs_map, model_answers, model_label="RAG-Baseline"):
    print("\n" + "=" * 60)
    print(f"  来源锚定评测 ({model_label})")
    print("=" * 60)

    judge = LLMJudge()
    results = judge.batch_evaluate(
        test_cases,
        retrieved_docs_map=retrieved_docs_map,
        model_answers=model_answers,
    )

    reporter = EvalReporter(results, model_label=model_label)
    report_path = reporter.save_report()
    results_path = reporter.save_results_json()

    print(f"\n✓ 评测报告: {report_path}")
    print(f"✓ 详细结果: {results_path}")

    scores = reporter.scores
    print(f"\n📊 评测摘要:")

    by_cat = scores.get("by_category", {})
    if "retrieval" in by_cat:
        r = by_cat["retrieval"]
        print(f"  检索: 机构命中率={r['org_hit_rate']:.0%} 关键词覆盖={r['avg_keyword_coverage']:.0%}")
    if "faithfulness" in by_cat:
        f = by_cat["faithfulness"]
        print(f"  忠实度: 判断准确率={f['accuracy']:.0%}")
    if "factual" in by_cat:
        fa = by_cat["factual"]
        print(f"  事实准确性: 准确率={fa['avg_accuracy']:.0%} 无幻觉率={fa['no_hallucination_rate']:.0%}")

    print(f"\n  综合评分: {scores.get('overall_score', 'N/A')}")
    return results


def main():
    parser = argparse.ArgumentParser(description="来源锚定金融LLM评测")
    parser.add_argument("--build-benchmark", action="store_true", help="构建Benchmark")
    parser.add_argument("--full", action="store_true", help="完整评测流程")
    parser.add_argument("--label", type=str, default="RAG-Baseline", help="模型标签")
    args = parser.parse_args()

    if args.full:
        test_cases = build_benchmark()
        retrieved_docs_map, model_answers = generate_rag_responses(test_cases)
        run_evaluation(test_cases, retrieved_docs_map, model_answers, model_label=args.label)
    elif args.build_benchmark:
        build_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
