"""
评测报告生成器 — 基于真实来源的评测分析与优化建议
"""
import json
import os
from datetime import datetime

from config import EVAL_DIR
from evaluation.metrics import compute_scores, dimension_breakdown, bad_case_analysis


class EvalReporter:
    """评测报告生成器"""

    def __init__(self, eval_results: list[dict], model_label: str = ""):
        self.results = eval_results
        self.model_label = model_label
        self.scores = compute_scores(eval_results) if eval_results else {}

    def generate_markdown(self) -> str:
        """生成Markdown评测报告"""
        s = self.scores

        lines = [
            f"# 金融LLM评测报告（来源锚定评测）",
            f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**评测模型**：{self.model_label or '未标注'}",
            f"**总评测样本**：{s.get('total_cases', 0)}",
            "",
            "---",
            "",
            f"## 一、综合评分：{s.get('overall_score', 'N/A')}",
            "",
            "本评测基于**真实研报来源**，不依赖LLM主观打分，所有指标均可追溯到原文。",
            "",
        ]

        by_cat = s.get("by_category", {})

        # 检索评测
        if "retrieval" in by_cat:
            r = by_cat["retrieval"]
            lines += [
                "## 二、检索评测",
                "",
                "衡量系统根据自然语言查询找到正确研报的能力。",
                "",
                f"| 指标 | 得分 |",
                f"|------|------|",
                f"| 样本数 | {r['count']} |",
                f"| 机构命中率 | **{r['org_hit_rate']:.0%}** |",
                f"| 关键词覆盖率 | **{r['avg_keyword_coverage']:.0%}** |",
                f"| 综合检索分 | **{r['avg_overall']:.2f}** |",
                "",
            ]
            if r["org_hit_rate"] < 0.5:
                lines.append("> ⚠️ 机构命中率偏低，建议优化Embedding模型或增加元数据索引。")

        # 忠实度评测
        if "faithfulness" in by_cat:
            f = by_cat["faithfulness"]
            lines += [
                "## 三、忠实度评测",
                "",
                "衡量模型判断「声明是否被原文支持」的准确率。",
                "",
                f"| 指标 | 得分 |",
                f"|------|------|",
                f"| 样本数 | {f['count']} |",
                f"| 判断准确率 | **{f['accuracy']:.0%}** |",
                "",
            ]
            if f["accuracy"] < 0.8:
                lines.append("> ⚠️ 忠实度判断准确率偏低，模型可能难以区分原文支持和曲解的内容。")

        # 事实准确性
        if "factual" in by_cat:
            fa = by_cat["factual"]
            lines += [
                "## 四、事实准确性评测",
                "",
                "衡量模型回答中关键事实与原文的一致性。",
                "",
                f"| 指标 | 得分 |",
                f"|------|------|",
                f"| 样本数 | {fa['count']} |",
                f"| 事实准确率 | **{fa['avg_accuracy']:.0%}** |",
                f"| 完全正确率 | **{fa['correct_rate']:.0%}** |",
                f"| 无幻觉率 | **{fa['no_hallucination_rate']:.0%}** |",
                "",
            ]
            if fa["no_hallucination_rate"] < 0.9:
                lines.append("> ⚠️ 存在幻觉问题，建议加强检索约束和数据验证机制。")

        # 优化建议
        lines += [
            "",
            "## 五、优化建议",
            "",
        ]
        suggestions = self._generate_suggestions()
        for i, sug in enumerate(suggestions):
            lines.append(f"{i+1}. **{sug['title']}**：{sug['detail']}")

        lines += [
            "",
            "---",
            "*本报告由金融LLM评测平台自动生成 · 所有指标可追溯到真实研报来源*",
        ]

        return "\n".join(lines)

    def _generate_suggestions(self) -> list[dict]:
        suggestions = []
        by_cat = self.scores.get("by_category", {})

        r = by_cat.get("retrieval", {})
        if r.get("org_hit_rate", 1.0) < 0.5:
            suggestions.append({
                "title": "提升检索精度",
                "detail": f"当前机构命中率仅{r['org_hit_rate']:.0%}。建议：1) 优化文档元数据索引，为每个chunk标注机构名；2) 尝试HyDE或Query改写提升检索语义匹配；3) 混合元数据过滤+语义检索。",
            })

        f = by_cat.get("faithfulness", {})
        if f.get("accuracy", 1.0) < 0.8:
            suggestions.append({
                "title": "增强忠实度判断能力",
                "detail": f"忠实度判断准确率{f['accuracy']:.0%}。建议：1) 在训练数据中增加'声明验证'类样本；2) 微调时加入NLI（自然语言推理）任务；3) 增加多证据交叉验证机制。",
            })

        fa = by_cat.get("factual", {})
        if fa.get("no_hallucination_rate", 1.0) < 0.9:
            suggestions.append({
                "title": "减少幻觉输出",
                "detail": f"幻觉率{1-fa['no_hallucination_rate']:.0%}。建议：1) 强化System Prompt中的'资料不足时明确告知'约束；2) 对涉及数字的回答增加后验证；3) 引入检索质量阈值，检索分过低时拒绝回答。",
            })

        if fa.get("avg_accuracy", 1.0) < 0.7:
            suggestions.append({
                "title": "提升事实准确性",
                "detail": f"事实准确率仅{fa['avg_accuracy']:.0%}。建议：1) 增加更多高质量研报数据；2) 优化检索结果Top-K和Chunk Size；3) 引入引用标注机制，强制模型在关键事实上引用来源。",
            })

        if not suggestions:
            suggestions.append({
                "title": "持续扩展评测场景",
                "detail": "当前模型在各维度表现良好。建议：1) 扩展更多金融子领域；2) 增加对抗性测试样本；3) 定期更新Benchmark以覆盖新的金融事件和政策。",
            })

        return suggestions

    def save_report(self, filename: str = "eval_report.md") -> str:
        path = os.path.join(EVAL_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())
        return path

    def save_results_json(self, filename: str = "eval_results.json") -> str:
        path = os.path.join(EVAL_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        return path
