"""
CLI — 构建真实金融研报训练语料
所有语料来自真实研报原文，经过清洗、分类、去重、质量过滤。

用法:
  python scripts/build_corpus.py                  # 全部场景，默认500条
  python scripts/build_corpus.py --max 300        # 指定条数
  python scripts/build_corpus.py --export my_corpus.jsonl
"""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
from corpus.builder import CorpusBuilder
from corpus.scenarios import SCENARIOS

parser = argparse.ArgumentParser(description="构建真实金融研报训练语料")
parser.add_argument("--scenario", type=str, default="all",
                    help=f"场景: {', '.join(s['name'] for s in SCENARIOS.values())} 或 all")
parser.add_argument("--max", type=int, default=500, dest="total",
                    help="目标语料条数（默认500）")
parser.add_argument("--export", type=str, default="corpus.jsonl",
                    help="输出文件名")
args = parser.parse_args()

# 解析场景
if args.scenario == "all":
    scenario_keys = list(SCENARIOS.keys())
else:
    name_to_key = {v["name"]: k for k, v in SCENARIOS.items()}
    key = name_to_key.get(args.scenario)
    if key is None:
        print(f"未知场景 '{args.scenario}'，可用: {list(name_to_key.keys())}")
        sys.exit(1)
    scenario_keys = [key]

print("=" * 60)
print("  真实金融研报训练语料构建")
print(f"  场景: {', '.join(SCENARIOS[k]['name'] for k in scenario_keys)}")
print(f"  目标: {args.total} 条")
print(f"  数据来源: 真实研报原文（清洗/分类/去重）")
print("=" * 60)

builder = CorpusBuilder()
builder.build(scenario_keys=scenario_keys, total_chunks=args.total)

stats = builder.stats()
print(f"\n{'='*60}")
print("📊 语料统计")
print(f"  总条数: {stats['total_chunks']}")
print(f"  总字符数: {stats['total_characters']:,}")
print(f"  平均每条: {stats['avg_chars_per_chunk']} 字符")
print(f"  场景分布:")
for scenario, count in stats.get("by_scenario", {}).items():
    print(f"    {scenario}: {count} 条")
if stats.get("top_orgs"):
    print(f"  主要来源机构: {list(stats['top_orgs'].keys())[:5]}")
if stats.get("top_stocks"):
    print(f"  主要覆盖股票: {list(stats['top_stocks'].keys())[:5]}")

path = builder.export(args.export)
print(f"\n✓ 语料已保存: {path}")
