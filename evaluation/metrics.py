"""
评测指标计算 — 基于真实来源的量化指标
"""
from collections import defaultdict
import statistics


def compute_scores(eval_results: list[dict]) -> dict:
    """
    聚合评测结果，按类别分别计算指标
    """
    if not eval_results:
        return {"error": "无评测数据"}

    # 按类别分组
    by_category = defaultdict(list)
    for r in eval_results:
        cat = r.get("category", "unknown")
        by_category[cat].append(r)

    result = {
        "total_cases": len(eval_results),
        "by_category": {},
    }

    # 检索评测指标
    retrieval_results = by_category.get("retrieval", [])
    if retrieval_results:
        org_hits = [r["scores"].get("org_hit", 0) for r in retrieval_results]
        kw_covers = [r["scores"].get("keyword_coverage", 0) for r in retrieval_results]
        overalls = [r["scores"].get("overall", 0) for r in retrieval_results]
        result["by_category"]["retrieval"] = {
            "count": len(retrieval_results),
            "org_hit_rate": round(statistics.mean(org_hits), 3),
            "avg_keyword_coverage": round(statistics.mean(kw_covers), 3),
            "avg_overall": round(statistics.mean(overalls), 3),
        }

    # 忠实度评测指标
    faithfulness_results = by_category.get("faithfulness", [])
    if faithfulness_results:
        corrects = [r["scores"].get("faithfulness_correct", 0) for r in faithfulness_results]
        result["by_category"]["faithfulness"] = {
            "count": len(faithfulness_results),
            "accuracy": round(statistics.mean(corrects), 3),
        }

    # 事实准确性评测指标
    factual_results = by_category.get("factual", [])
    if factual_results:
        accuracies = [r["scores"].get("factual_accuracy", 0) for r in factual_results]
        is_correct = [r["scores"].get("is_correct", 0) for r in factual_results]
        no_hallu = [r["scores"].get("hallucination", 0) for r in factual_results]
        result["by_category"]["factual"] = {
            "count": len(factual_results),
            "avg_accuracy": round(statistics.mean(accuracies), 3),
            "correct_rate": round(statistics.mean(is_correct), 3),
            "no_hallucination_rate": round(statistics.mean(no_hallu), 3),
        }

    # 综合分数（加权平均）
    weights = {"retrieval": 0.3, "faithfulness": 0.3, "factual": 0.4}
    overall = 0
    total_weight = 0
    for cat, info in result["by_category"].items():
        w = weights.get(cat, 0.25)
        if cat == "retrieval":
            overall += info["avg_overall"] * w
        elif cat == "faithfulness":
            overall += info["accuracy"] * w
        elif cat == "factual":
            overall += info["avg_accuracy"] * w
        total_weight += w

    result["overall_score"] = round(overall / max(0.001, total_weight), 3)

    return result


def dimension_breakdown(eval_results: list[dict], group_by: str = "category") -> dict:
    """按类别/维度分组统计"""
    groups = defaultdict(list)
    for r in eval_results:
        key = r.get(group_by, r.get("category", "unknown"))
        scores = r.get("scores", {})
        overall = scores.get("overall", scores.get("factual_accuracy", scores.get("faithfulness_correct", 0)))
        groups[key].append(overall)

    breakdown = {}
    for key, s_list in groups.items():
        breakdown[key] = {
            "count": len(s_list),
            "mean": round(statistics.mean(s_list), 3),
            "std": round(statistics.stdev(s_list), 3) if len(s_list) > 1 else 0,
        }
    return breakdown


def bad_case_analysis(eval_results: list[dict], threshold: float = 0.5) -> list[dict]:
    """识别低分case"""
    bad = []
    for r in eval_results:
        scores = r.get("scores", {})
        overall = scores.get("overall", scores.get("factual_accuracy", scores.get("faithfulness_correct", 0)))
        if overall < threshold:
            r["_overall"] = round(overall, 3)
            bad.append(r)
    return sorted(bad, key=lambda x: x["_overall"])
