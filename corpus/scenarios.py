"""
金融领域训练语料场景定义
覆盖JD要求的四大场景：投研分析、金融问答、财报解读、合规咨询
"""

SCENARIOS = {
    "investment_research": {
        "name": "投研分析",
        "description": "行业趋势、个股研究、策略分析、产业链调研",
        "prompt_prefix": "你是一位资深行业研究员。",
    },
    "financial_qa": {
        "name": "金融问答",
        "description": "金融概念解释、市场机制、投资知识、产品对比",
        "prompt_prefix": "你是一位金融知识专家。",
    },
    "report_interpretation": {
        "name": "财报解读",
        "description": "财务报表分析、指标解读、业绩归因、估值分析",
        "prompt_prefix": "你是一位财务分析专家。",
    },
    "compliance": {
        "name": "合规咨询",
        "description": "监管政策、合规要求、信息披露、风险管理",
        "prompt_prefix": "你是一位合规咨询顾问。",
    },
}

QUESTION_TYPES = [
    {
        "type": "factual",
        "name": "事实查询",
        "template": "根据材料直接提取事实信息，答案明确、无歧义。",
    },
    {
        "type": "analytical",
        "name": "分析推理",
        "template": "需要结合材料中的多个信息点进行推理分析，给出有逻辑的结论。",
    },
    {
        "type": "comparative",
        "name": "对比分析",
        "template": "比较不同标的、不同时期或不同观点，指出差异与共性。",
    },
    {
        "type": "predictive",
        "name": "趋势判断",
        "template": "基于材料中的数据和信息，对行业或公司的未来趋势进行有依据的判断。",
    },
]

# 评估维度定义
EVAL_DIMENSIONS = [
    {
        "key": "accuracy",
        "name": "准确性",
        "weight": 0.35,
        "description": "回答是否基于材料事实，数据引用是否正确",
        "rubric": {
            "5": "完全准确，所有数据和事实均可在材料中找到出处",
            "3": "大部分准确，偶有表述模糊但不影响核心结论",
            "1": "存在明显事实错误或编造信息",
        },
    },
    {
        "key": "professionalism",
        "name": "专业性",
        "weight": 0.25,
        "description": "是否使用规范的金融术语，分析框架是否专业",
        "rubric": {
            "5": "术语使用规范，分析框架符合行业标准，逻辑严密",
            "3": "基本专业，但术语使用偶有不规范或分析深度不足",
            "1": "概念混淆，术语使用错误，分析逻辑混乱",
        },
    },
    {
        "key": "compliance",
        "name": "合规性",
        "weight": 0.20,
        "description": "是否包含风险提示，有无投资建议合规问题",
        "rubric": {
            "5": "包含必要的风险提示和免责声明，无违规投资建议",
            "3": "有基本的风险意识但不够充分",
            "1": "缺少风险提示，或存在误导性投资建议",
        },
    },
    {
        "key": "completeness",
        "name": "完整性",
        "weight": 0.20,
        "description": "是否全面回答了问题，有无遗漏关键信息",
        "rubric": {
            "5": "全面覆盖问题的所有方面，无关键信息遗漏",
            "3": "覆盖了主要方面，但有部分细节缺失",
            "1": "遗漏多个关键信息点，回答不完整",
        },
    },
]
