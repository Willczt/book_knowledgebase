"""用评测集校准「可信分阈值」：正负样本分数分布 + 阈值扫描。

问题：`min_rerank_score` 设多少才算「低于它不可信」？
答：不能拍脑袋，要用带标注的评测集统计两件事——

  1) 正样本（gold 片段）的 rerank 分分布：如果 10% 的正样本只有 0.2 分，
     阈值设 0.3 就会漏答 10% 的问题；
  2) 负样本（非 gold 片段）的分布：阈值越高，噪音剔得越干净。

注意局限：评测集只标了「哪些是答案」，没标「哪些一定无关」。所以
"精确率"不可信（未标注 ≠ 无关），只看 gold 保留率与噪音剔除率。

用法：
    .venv/bin/python scripts/calibrate_threshold.py \
        --dataset eval/dataset.jsonl --kb-id <kb> --base-url http://127.0.0.1:8001

输出：eval/reports/threshold_calibration_<ts>.md/.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent

GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("gold_chunk_ids"):
            rows.append(row)
    return rows


def pctl(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * ratio
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 4)


async def fetch(
    client: httpx.AsyncClient, base_url: str, kb_id: str, question: str, sem: asyncio.Semaphore
) -> list[dict[str, Any]]:
    async with sem:
        for attempt in range(2):
            try:
                resp = await client.post(
                    f"{base_url}/api/v1/knowledge-bases/{kb_id}/search",
                    json={
                        "query": question,
                        "top_k": 30,
                        "score_threshold": 0.0,
                        "min_rerank_score": 0.0,  # 关键：标定时不做任何过滤
                        "use_reranker": True,
                    },
                )
                resp.raise_for_status()
                return resp.json().get("results") or []
            except Exception as exc:
                if attempt == 1:
                    print(f"  [warn] 请求失败：{type(exc).__name__}: {exc}")
                    return []
                await asyncio.sleep(1)
    return []


async def main_async(args: argparse.Namespace) -> None:
    cases = load_cases(ROOT / args.dataset)
    print(f"数据集：{len(cases)} 条（只取有 gold 的）")

    sem = asyncio.Semaphore(args.concurrency)
    positive: list[float] = []   # gold 片段的 rerank 分
    negative: list[float] = []   # 非 gold 片段的 rerank 分
    per_case: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=180) as client:
        tasks = [
            fetch(client, args.base_url, args.kb_id, case["question"], sem) for case in cases
        ]
        results = await asyncio.gather(*tasks)

    for case, items in zip(cases, results):
        if not items:
            continue
        gold = set(case["gold_chunk_ids"])
        rows = [
            {"chunk_id": item["chunk_id"], "score": float(item["score"]), "is_gold": item["chunk_id"] in gold}
            for item in items
        ]
        for row in rows:
            (positive if row["is_gold"] else negative).append(row["score"])
        per_case.append(
            {
                "id": case["id"],
                "type": case.get("type"),
                "gold_scores": sorted([r["score"] for r in rows if r["is_gold"]], reverse=True),
                "max_non_gold": max([r["score"] for r in rows if not r["is_gold"]], default=0.0),
                "rows": rows,
            }
        )

    if not positive:
        raise SystemExit("没有收集到正样本：确认 rerank 服务可用、gold id 与集合一致")

    print(f"正样本 {len(positive)} 条（gold），负样本 {len(negative)} 条")
    print("正样本分位：", {q: pctl(positive, q) for q in (0.05, 0.1, 0.25, 0.5, 0.75)})
    print("负样本分位：", {q: pctl(negative, q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)})

    grid_rows = []
    for threshold in GRID:
        kept_gold = sum(1 for score in positive if score >= threshold)
        kept_gold_cases = sum(
            1 for case in per_case if any(score >= threshold for score in case["gold_scores"])
        )
        dropped_neg = sum(1 for score in negative if score < threshold)
        kept_total = sum(1 for case in per_case for row in case["rows"] if row["score"] >= threshold)
        grid_rows.append(
            {
                "threshold": threshold,
                "gold_keep_rate": round(kept_gold / len(positive), 4),
                "case_answerable_rate": round(kept_gold_cases / len(per_case), 4),
                "noise_removal_rate": round(dropped_neg / len(negative), 4),
                "avg_kept_per_query": round(kept_total / len(per_case), 2),
            }
        )

    # 两个工作点，让业务按「漏答 vs 噪音」的代价自己选：
    #   保守：gold 保留率 ≥ 97%（几乎不误杀，主要用来省上下文）
    #   激进：gold 保留率 ≥ 94%（多剔噪音，接受少量漏答）
    conservative_pool = [row for row in grid_rows if row["gold_keep_rate"] >= 0.97]
    aggressive_pool = [row for row in grid_rows if row["gold_keep_rate"] >= 0.94]
    conservative = (
        max(conservative_pool, key=lambda r: r["noise_removal_rate"])
        if conservative_pool
        else grid_rows[0]
    )
    aggressive = (
        max(aggressive_pool, key=lambda r: r["noise_removal_rate"])
        if aggressive_pool
        else grid_rows[-1]
    )

    lines = [
        "# 可信分阈值标定",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 数据集：{args.dataset}（{len(per_case)} 条有效）",
        f"- 正样本 {len(positive)} / 负样本 {len(negative)}",
        "",
        "## 分数分布（rerank 分）",
        "",
        "| 分位 | P5 | P10 | P25 | P50 | P75 |",
        "|---|---|---|---|---|---|",
        f"| 正样本 | {pctl(positive,0.05)} | {pctl(positive,0.1)} | {pctl(positive,0.25)} | {pctl(positive,0.5)} | {pctl(positive,0.75)} |",
        "",
        "| 分位 | P50 | P75 | P90 | P95 | P99 |",
        "|---|---|---|---|---|---|",
        f"| 负样本 | {pctl(negative,0.5)} | {pctl(negative,0.75)} | {pctl(negative,0.9)} | {pctl(negative,0.95)} | {pctl(negative,0.99)} |",
        "",
        "## 阈值扫描",
        "",
        "| 阈值 | gold 保留率 | 问题可答率 | 噪音剔除率 | 平均保留条数 |",
        "|---|---|---|---|---|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['gold_keep_rate']:.3f} | {row['case_answerable_rate']:.3f} | "
            f"{row['noise_removal_rate']:.3f} | {row['avg_kept_per_query']:.2f} |"
        )
    lines += [
        "",
        "## 工作点建议",
        "",
        f"- **保守（几乎不误杀）**：`min_rerank_score = {conservative['threshold']:.2f}` — "
        f"gold 保留 {conservative['gold_keep_rate']:.1%}，噪音剔除 {conservative['noise_removal_rate']:.1%}，"
        f"平均保留 {conservative['avg_kept_per_query']:.1f} 条",
        f"- **激进（多剔噪音）**：`min_rerank_score = {aggressive['threshold']:.2f}` — "
        f"gold 保留 {aggressive['gold_keep_rate']:.1%}，噪音剔除 {aggressive['noise_removal_rate']:.1%}，"
        f"平均保留 {aggressive['avg_kept_per_query']:.1f} 条",
        "",
        "> 口径说明：评测集只标注了 gold 片段，未标注「一定无关」，所以不报精确率；",
        "> 正负样本分数分布重叠严重时，**不要用单阈值做「不可信就拒答」**，",
        "> 它只能用来降噪/省上下文；判「不可信」需要额外信号（证据一致性、模型自评、跨文档佐证）。",
    ]

    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"threshold_calibration_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "positive_n": len(positive),
                "negative_n": len(negative),
                "positive_pct": {str(q): pctl(positive, q) for q in (0.05, 0.1, 0.25, 0.5, 0.75)},
                "negative_pct": {str(q): pctl(negative, q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                "grid": grid_rows,
                "recommended_conservative": conservative,
                "recommended_aggressive": aggressive,
                "per_case": per_case,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + "\n".join(lines))
    print(f"\n报告：{out_dir / f'{stem}.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="校准精排可信分阈值")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
