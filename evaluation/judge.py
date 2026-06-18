"""
评测裁判 — 基于真实来源的量化评测
不依赖LLM主观打分，用可验证的指标衡量模型表现：
- 检索命中率：检索到的文档是否包含正确答案
- 答案忠实度：回答内容是否可追溯到来源文档
- 事实准确性：回答中的关键事实是否与来源一致
"""
import json
import time
import re
from datetime import datetime
from collections import defaultdict

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMJudge:
    """来源锚定评测器"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # ─── 检索评测 ───
    def evaluate_retrieval(
        self, query: str, retrieved_docs: list[dict], expected_org: str, expected_keywords: list[str]
    ) -> dict:
        """
        评测检索质量
        指标: 机构命中、关键词覆盖、排名位置
        """
        org_hit = False
        org_rank = -1
        keyword_hits = 0
        keyword_total = len(expected_keywords) if expected_keywords else 1

        for i, doc in enumerate(retrieved_docs):
            doc_org = doc.get("org", "")
            doc_text = doc.get("content", "")

            # 机构命中
            if expected_org and expected_org in doc_org:
                if not org_hit:
                    org_hit = True
                    org_rank = i + 1

        # 关键词覆盖
        all_text = " ".join(d.get("content", "") for d in retrieved_docs)
        for kw in expected_keywords:
            if kw in all_text:
                keyword_hits += 1

        score = (
            (1.0 if org_hit else 0.0) * 0.5
            + (keyword_hits / keyword_total) * 0.3
            + (1.0 / max(1, org_rank) if org_hit else 0.0) * 0.2
        )

        return {
            "query": query,
            "scores": {
                "org_hit": 1.0 if org_hit else 0.0,
                "keyword_coverage": round(keyword_hits / keyword_total, 2),
                "rank_score": round(1.0 / max(1, org_rank), 2) if org_hit else 0.0,
                "overall": round(score, 2),
            },
            "details": {
                "org_rank": org_rank,
                "keywords_found": keyword_hits,
                "keywords_total": keyword_total,
                "retrieved_orgs": list(set(d.get("org", "") for d in retrieved_docs[:5])),
            },
        }

    # ─── 忠实度评测 ───
    def evaluate_faithfulness(self, claim: str, source_text: str, model_judgment: str) -> dict:
        """
        评测模型对声明忠实度的判断
        model_judgment: 模型对"此声明是否被原文支持"的判断
        """
        # 用LLM做结构化判断验证
        prompt = f"""判断以下模型回答是否正确识别了声明的真实性。

原文：
{source_text[:1500]}

声明：{claim}

模型判断：{model_judgment}

请分析模型判断是否正确。输出JSON：
```json
{{
  "model_correct": true/false,
  "explanation": "一句话解释",
  "correct_judgment": "supported或not_supported"
}}
```"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            result = json.loads(self._extract_json(text))
            return {
                "claim": claim,
                "scores": {
                    "faithfulness_correct": 1.0 if result.get("model_correct") else 0.0,
                },
                "details": result,
            }
        except:
            return {
                "claim": claim,
                "scores": {"faithfulness_correct": 0.0},
                "details": {"error": "评测失败"},
            }

    # ─── 事实准确性评测 ───
    def evaluate_factual(self, question: str, expected_answer: str, source_text: str, model_answer: str) -> dict:
        """
        评测模型回答的事实准确性
        对比模型回答与标准答案+原文
        """
        prompt = f"""对比模型回答与标准答案，评估事实准确性。

问题：{question}

标准答案（来自原文）：{expected_answer}

原文依据：{source_text[:1200]}

模型回答：{model_answer}

评分标准：
- 5: 回答与标准答案一致，关键数字/事实完全正确
- 4: 基本正确，有轻微表述差异但不影响理解
- 3: 部分正确，有遗漏或模糊
- 2: 有事实错误或关键信息遗漏
- 1: 完全错误或编造

输出JSON：
```json
{{
  "factual_score": 1-5,
  "is_correct": true/false,
  "key_differences": "主要差异",
  "hallucination_detected": true/false
}}
```"""

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            result = json.loads(self._extract_json(text))
            return {
                "question": question,
                "scores": {
                    "factual_accuracy": result.get("factual_score", 0) / 5.0,
                    "is_correct": 1.0 if result.get("is_correct") else 0.0,
                    "hallucination": 0.0 if result.get("hallucination_detected") else 1.0,
                },
                "details": result,
            }
        except:
            return {
                "question": question,
                "scores": {"factual_accuracy": 0.0, "is_correct": 0.0, "hallucination": 0.0},
                "details": {"error": "评测失败"},
            }

    # ─── 批量评测 ───
    def batch_evaluate(
        self,
        test_cases: list[dict],
        retrieved_docs_map: dict[str, list[dict]] | None = None,
        model_answers: dict[str, str] | None = None,
    ) -> list[dict]:
        """批量评测，根据test case类型自动选择评测方法"""
        results = []
        total = len(test_cases)

        for i, case in enumerate(test_cases):
            cat = case.get("category", "")
            case_id = case.get("id", str(i))
            print(f"  评测 [{i+1}/{total}] {cat}: {str(case.get('query', case.get('claim', case.get('question', '')))[:60])}")

            if cat == "retrieval":
                query = case.get("query", "")
                docs = (retrieved_docs_map or {}).get(query, [])
                result = self.evaluate_retrieval(
                    query=query,
                    retrieved_docs=docs,
                    expected_org=case.get("expected_org", ""),
                    expected_keywords=case.get("expected_keywords", []),
                )

            elif cat == "faithfulness":
                claim = case.get("claim", "")
                source = case.get("source_content", "")
                model_judgment = (model_answers or {}).get(case_id, "")
                result = self.evaluate_faithfulness(
                    claim=claim,
                    source_text=source,
                    model_judgment=model_judgment,
                )

            elif cat == "factual":
                question = case.get("question", "")
                expected = case.get("answer", "")
                source = case.get("source_content", "")
                model_answer = (model_answers or {}).get(case_id, "")
                result = self.evaluate_factual(
                    question=question,
                    expected_answer=expected,
                    source_text=source,
                    model_answer=model_answer,
                )

            else:
                continue

            result["test_case_id"] = case_id
            result["category"] = cat
            result["evaluated_at"] = datetime.now().isoformat()
            results.append(result)
            time.sleep(0.2)

        return results

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            s = text.index("```json") + 7
            e = text.index("```", s)
            return text[s:e].strip()
        if "```" in text:
            s = text.index("```") + 3
            e = text.index("```", s)
            return text[s:e].strip()
        if "{" in text and "}" in text:
            return text[text.index("{"):text.rindex("}") + 1]
        return text.strip()
