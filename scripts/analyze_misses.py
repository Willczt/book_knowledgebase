"""把"没召回"拆开看：是数据问题还是检索问题。

只看汇总指标无法决定优化方向：
  - Recall@30 掉点，可能是「gold 根本不在库里」（解析/分块/评测集过期）
  - 也可能是「在库里但两路召回都没进前 30」（embedding / BM25 / 查询表达）

用法：
    .venv/bin/python scripts/analyze_misses.py \
        --report eval/reports/ci_xxx.json --collection kb_xxx --kb-id <kb>

    # 按题型看、只看某一类
    ... --type multi_hop --only missed

输出：控制台摘要 + eval/reports/misses_<ts>.md
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent


def latest_report() -> Path:
    files = sorted(glob.glob(str(ROOT / "eval" / "reports" / "*.json")))
    if not files:
        raise SystemExit("eval/reports 下没有 JSON 报告，先跑一次 eval_ci.py")
    return Path(files[-1])


def load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    # 兼容：eval_ci 报告在 retrieval.off，eval_rag 报告在 reports[0]
    if "retrieval" in data:
        rows = data["retrieval"]["off"]["rows"]
    else:
        rows = next(r for r in data["reports"] if r["stage"] == "retrieval")["rows"]
    return rows, meta


def classify(
    rows: list[dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
    collection: str,
    host: str,
    port: int,
    top_k: int,
) -> dict[str, Any]:
    client = QdrantClient(host=host, port=port, timeout=30)
    gold_ids = sorted(
        {cid for row in rows for cid in dataset.get(row["id"], {}).get("gold_chunk_ids", [])}
    )
    found: set[str] = set()
    for start in range(0, len(gold_ids), 256):
        points = client.retrieve(
            collection_name=collection, ids=gold_ids[start : start + 256], with_payload=False
        )
        found |= {str(point.id) for point in points}

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gold = dataset.get(row["id"], {}).get("gold_chunk_ids", [])
        rank = row.get("gold_rank")
        if rank is not None and rank <= top_k:
            buckets["hit"].append(row)
            continue
        missing = [cid for cid in gold if cid not in found]
        if missing:
            buckets["not_indexed"].append({**row, "missing_gold": missing})
        else:
            buckets["missed"].append(row)
    return {"buckets": buckets, "gold_total": len(gold_ids), "gold_found": len(gold_ids) - (len(gold_ids) - len(found))}


def render(report: dict[str, Any], rows: list[dict[str, Any]], dataset: dict[str, dict[str, Any]]) -> str:
    buckets = report["buckets"]
    total = sum(len(v) for v in buckets.values())
    lines = [
        "# 未召回归因",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 样本：{total} 条",
        "",
        "## 分类",
        "",
        "| 类别 | 数量 | 含义 | 该改什么 |",
        "|---|---|---|---|",
        f"| 已命中 | {len(buckets['hit'])} | gold 进了前 K | — |",
        f"| 库里没有 | {len(buckets['not_indexed'])} | gold chunk 不在 Qdrant | 解析/分块/评测集过期 |",
        f"| 有但没召回 | {len(buckets['missed'])} | 在库里，但没进前 K | 查询表达 / embedding / 稀疏 / 融合 |",
        "",
    ]

    by_type: dict[str, Counter] = defaultdict(Counter)
    for name, items in buckets.items():
        for row in items:
            by_type[row.get("type", "?")][name] += 1

    if by_type:
        lines += ["## 按题型", "", "| 题型 | 已命中 | 库里没有 | 有但没召回 |", "|---|---|---|---|"]
        for qtype, counter in sorted(by_type.items()):
            lines.append(
                f"| {qtype} | {counter['hit']} | {counter['not_indexed']} | {counter['missed']} |"
            )
        lines.append("")

    for name, title in (("not_indexed", "库里没有（数据问题）"), ("missed", "有但没召回（检索问题）")):
        items = buckets[name]
        if not items:
            continue
        lines += [f"## {title}", ""]
        for row in items[:20]:
            meta = dataset.get(row["id"], {})
            lines.append(f"- `{row['id']}`（{row.get('type')}）{meta.get('question', '')}")
            if name == "not_indexed":
                lines.append(f"  - 缺失 gold：`{', '.join(row['missing_gold'][:2])}`")
                lines.append(f"  - 期望文档：{meta.get('source_doc')}")
                lines.append("  - → 检查该文档是否重新摄入过（chunk id 会变），或解析是否漏段")
            else:
                rank = row.get("gold_rank")
                lines.append(f"  - gold 排名：{rank if rank else '> ' + str(len(row.get('ranked') or []))}")
                lines.append(f"  - 期望文档：{meta.get('source_doc')}")
                lines.append("  - → 看改写后的 query 是否偏离；再试多查询 / HyDE / 调稀疏权重")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="未召回归因")
    parser.add_argument("--report", default="", help="报告 JSON（默认取最新）")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--kb-id", default="")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--type", default="", help="只看某个题型")
    parser.add_argument("--only", choices=["hit", "missed", "not_indexed"], default="")
    args = parser.parse_args()

    report_path = Path(args.report) if args.report else latest_report()
    rows, _ = load_rows(report_path)
    if args.type:
        rows = [row for row in rows if row.get("type") == args.type]
    if not rows:
        raise SystemExit("报告里没有符合条件的样本（旧报告可能没存完整排名，需重跑）")

    dataset = {}
    for line in (ROOT / args.dataset).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            dataset[item["id"]] = item

    result = classify(rows, dataset, args.collection, args.qdrant_host, args.qdrant_port, args.top_k)
    if args.only:
        result["buckets"] = {args.only: result["buckets"].get(args.only, [])}

    markdown = render(result, rows, dataset)
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"misses_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n报告：{out_dir / f'{stem}.md'}")


if __name__ == "__main__":
    main()
