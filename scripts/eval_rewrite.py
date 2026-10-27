"""多轮追问的查询改写评测：同一条依赖上文的追问，分别用 off / concat / llm 生成检索查询，
比较召回指标的差距。

为什么单独评：`eval_rag.py` 里每条问题都自带实体、可独立检索，改写根本不会触发，
混在一起评不出改写的价值。这里用 `eval/multiturn.jsonl`：每题给一段上文 + 一句
只含指代/省略的追问，gold 片段沿用 `eval/dataset.jsonl` 里同源问题的标注。

三种策略：
    off     不做改写，直接拿追问去检索（多轮体验的现状）
    concat  把上一轮提问和追问拼起来（不调模型，改写失败时的兜底）
    llm     用 LLM 补全指代（生产默认）

用法（仓库根目录，后端服务需已启动）：
    .venv/bin/python scripts/eval_rewrite.py --kb-id <kb_id>
    .venv/bin/python scripts/eval_rewrite.py --kb-id <kb_id> --dry-run   # 只看改写结果
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_rag import hit_at_1, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402

from config import get_settings  # noqa: E402
from core.llm.base import ChatMessage, MessageRole  # noqa: E402
from core.llm.registry import create_llm_provider  # noqa: E402
from core.rag import query_rewriter as qr  # noqa: E402

MODES = ("off", "concat", "llm", "hybrid")
_METRIC_KEYS = ("hit@1", "recall@5", "recall@10", "mrr@10")


@dataclass
class Case:
    id: str
    question: str
    type: str
    gold_chunk_ids: list[str]
    history: list[ChatMessage]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def load_cases(dataset_path: Path, multiturn_path: Path) -> tuple[list[Case], str]:
    gold: dict[str, list[str]] = {}
    kb_id = ""
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold[row["id"]] = row.get("gold_chunk_ids") or []
            kb_id = kb_id or (row.get("kb_id") or "")

    cases: list[Case] = []
    with multiturn_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            base = row.get("base_case")
            if base and not gold.get(base):
                raise SystemExit(f"{row['id']}: 找不到 {base} 的 gold，评测集可能已改版")
            inline_gold = row.get("gold_chunk_ids") or []
            cases.append(
                Case(
                    id=row["id"],
                    question=row["question"],
                    type=row.get("type") or "pronoun",
                    gold_chunk_ids=inline_gold or gold.get(base, []),
                    history=[
                        ChatMessage(role=MessageRole(m["role"]), content=m["content"])
                        for m in row.get("history") or []
                    ],
                )
            )
    if not cases:
        raise SystemExit(f"没有可用用例：{multiturn_path}")
    return cases, kb_id


def build_llm(provider: str, model_id: str):
    return create_llm_provider(
        provider,
        model_id,
        {
            "max_tokens": qr.settings.query_rewrite_max_tokens,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "timeout": 60,
        },
    )


async def search(
    client: httpx.AsyncClient, base_url: str, kb_id: str, query: str, top_k: int
) -> tuple[list[str], float]:
    body = {"query": query, "top_k": top_k, "score_threshold": 0.0, "min_rerank_score": 0.0}
    started = time.perf_counter()
    resp = await client.post(f"{base_url}/api/v1/knowledge-bases/{kb_id}/search", json=body)
    resp.raise_for_status()
    latency = (time.perf_counter() - started) * 1000
    results = resp.json().get("results") or []
    return [item["chunk_id"] for item in results], round(latency, 1)


async def run_mode(
    mode: str,
    cases: list[Case],
    base_url: str,
    kb_id: str,
    top_k: int,
    llm: Any,
    dry_run: bool,
) -> dict[str, Any]:
    # rewrite_query 读模块级 settings，改这一处即可切换策略（与生产同一份代码）
    # hybrid = LLM 改写后再拼上一轮提问（回放型基线：两种信号都给检索）
    qr.settings.query_rewrite = "llm" if mode == "hybrid" else mode
    rows: list[dict[str, Any]] = []
    rewrite_costs: list[float] = []

    async with httpx.AsyncClient(timeout=180) as client:
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            query, method = await qr.rewrite_query(
                llm if mode in ("llm", "hybrid") else None, case.history, case.question
            )
            if mode == "hybrid" and method == "llm":
                previous = [
                    m.content
                    for m in case.history
                    if m.role == MessageRole.user and isinstance(m.content, str)
                ]
                if previous:
                    method = "hybrid"
                    query = f"{previous[-1].strip()} {query}"
            rewrite_ms = round((time.perf_counter() - started) * 1000, 1)
            rewrite_costs.append(rewrite_ms)

            row: dict[str, Any] = {
                "id": case.id,
                "type": case.type,
                "question": case.question,
                "retrieval_query": query,
                "method": method,
                "rewrite_ms": rewrite_ms,
            }
            if dry_run:
                rank: int | None = None
            else:
                try:
                    ranked, search_ms = await search(client, base_url, kb_id, query, top_k)
                except Exception as exc:  # 单条失败不拖垮整轮
                    print(f"  [{mode}] {case.id} 检索失败：{type(exc).__name__}: {exc}")
                    row["error"] = str(exc)
                    rows.append(row)
                    continue
                gold = set(case.gold_chunk_ids)
                rank = next((i for i, cid in enumerate(ranked, 1) if cid in gold), None)
                row.update(
                    {
                        "rank": rank,
                        "search_ms": search_ms,
                        "hit@1": hit_at_1(ranked, case.gold_chunk_ids),
                        "recall@5": recall_at_k(ranked, case.gold_chunk_ids, 5),
                        "recall@10": recall_at_k(ranked, case.gold_chunk_ids, 10),
                        "mrr@10": mrr_at_k(ranked, case.gold_chunk_ids, 10),
                        "ndcg@10": ndcg_at_k(ranked, case.gold_chunk_ids, 10),
                    }
                )
            rows.append(row)
            print(
                f"  [{mode}] {index:>2}/{len(cases)} rank={str(rank or '-'):>3} "
                f"{method:<6} {query[:46]}"
            )

    scored = [row for row in rows if "error" not in row and "rank" in row]
    summary = {
        key: _mean([row[key] for row in scored]) for key in (*_METRIC_KEYS, "ndcg@10")
    }
    summary["n"] = len(scored)
    summary["rewrite_ms_p50"] = round(sorted(rewrite_costs)[len(rewrite_costs) // 2], 1)
    summary["rewrite_ms_mean"] = round(_mean(rewrite_costs), 1)
    summary["retrieval_rate"] = _mean([1.0 if row.get("rank") else 0.0 for row in scored])
    return {"mode": mode, "summary": summary, "rows": rows}


def render(meta: dict[str, Any], reports: list[dict[str, Any]], dry_run: bool) -> str:
    lines = ["# 多轮追问改写评测", ""]
    lines += [
        f"- 时间：{meta['time']}",
        f"- 知识库：`{meta['kb_id']}`",
        f"- 用例：{meta['n_cases']} 条（`{meta['multiturn']}`，gold 来自 `{meta['dataset']}`）",
        f"- 后端：{meta['base_url']} · top_k={meta['top_k']}",
        "",
    ]
    if dry_run:
        lines += ["## 改写结果（未检索）", "", "| id | 追问 | 策略 | 方法 | 检索查询 |", "|---|---|---|---|---|"]
        for report in reports:
            for row in report["rows"]:
                lines.append(
                    f"| {row['id']} | {row['question']} | {report['mode']} | {row['method']} | "
                    f"{row['retrieval_query']} |"
                )
        return "\n".join(lines) + "\n"

    lines += ["## 汇总", "", "| 策略 | n | Hit@1 | R@5 | R@10 | MRR@10 | NDCG@10 | 检索命中率 | 改写耗时 P50 |", "|---|---|---|---|---|---|---|---|---|"]
    for report in reports:
        s = report["summary"]
        lines.append(
            f"| {report['mode']} | {s['n']} | {s['hit@1']:.3f} | {s['recall@5']:.3f} | "
            f"{s['recall@10']:.3f} | {s['mrr@10']:.3f} | {s['ndcg@10']:.3f} | "
            f"{s['retrieval_rate']:.3f} | {s['rewrite_ms_p50']} ms |"
        )

    by_mode = {report["mode"]: report["summary"] for report in reports}
    if "off" in by_mode:
        modes = [m for m in MODES if m in by_mode]
        lines += [
            "",
            "## 各策略对比（基准 off）",
            "",
            "| 指标 | " + " | ".join(modes) + " |",
            "|---|" + "---|" * len(modes),
        ]
        for key in _METRIC_KEYS:
            lines.append(
                "| " + key + " | " + " | ".join(f"{by_mode[m][key]:.3f}" for m in modes) + " |"
            )

    ranks: dict[str, dict[str, Any]] = {}
    for report in reports:
        for row in report["rows"]:
            ranks.setdefault(row["id"], {"question": row["question"]})[report["mode"]] = row
    modes = [m for m in MODES if m in by_mode]
    lines += [
        "",
        "## 逐题（数字为 gold 片段名次，✗ 表示 top_k 内没召回到）",
        "",
        "| id | 追问 | " + " | ".join(modes) + " | LLM 改写结果 |",
        "|---|" + "---|" * (len(modes) + 1),
    ]
    for case_id, item in ranks.items():
        cells = []
        for mode in modes:
            row = item.get(mode) or {}
            cells.append("-" if "rank" not in row else str(row["rank"] or "✗"))
        llm_row = item.get("llm") or {}
        lines.append(
            f"| {case_id} | {item['question'][:36]} | " + " | ".join(cells) + " | "
            f"{llm_row.get('retrieval_query', '')[:60]} |"
        )
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    cases, dataset_kb = load_cases(ROOT / args.dataset, ROOT / args.multiturn)
    kb_id = args.kb_id or dataset_kb
    if not kb_id:
        raise SystemExit("需要 --kb-id（或数据集里带 kb_id）")
    print(f"用例 {len(cases)} 条，kb_id={kb_id}")

    llm = None
    if "llm" in args.modes:
        model = args.llm_model or settings.deepseek_model
        try:
            llm = build_llm(args.llm_provider, model)
            print(f"改写模型：{args.llm_provider}/{model}")
        except Exception as exc:
            print(f"  [warn] 改写模型初始化失败，llm 策略将退化为拼接：{type(exc).__name__}: {exc}")

    reports = []
    for mode in args.modes:
        print(f"\n=== 策略 {mode} ===")
        reports.append(
            await run_mode(mode, cases, args.base_url, kb_id, args.top_k, llm, args.dry_run)
        )

    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "kb_id": kb_id,
        "dataset": args.dataset,
        "multiturn": args.multiturn,
        "n_cases": len(cases),
        "base_url": args.base_url,
        "top_k": args.top_k,
    }
    markdown = render(meta, reports, args.dry_run)
    print("\n" + "=" * 70)
    if not args.dry_run:
        by_mode = {report["mode"]: report["summary"] for report in reports}
        if "off" in by_mode:
            base = by_mode["off"]
            print("相对不改写（off）：")
            for mode, summary in by_mode.items():
                if mode == "off":
                    continue
                deltas = "  ".join(
                    f"{key} {summary[key] - base[key]:+.3f}" for key in _METRIC_KEYS
                )
                print(f"  {mode:8} {deltas}")
    print(markdown)

    if not args.dry_run:
        out_dir = ROOT / "eval" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rewrite_{args.label + '_' if args.label else ''}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
        (out_dir / f"{stem}.json").write_text(
            json.dumps({"meta": meta, "reports": reports}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"报告已写入 {out_dir / f'{stem}.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮追问的查询改写评测")
    parser.add_argument("--dataset", default="eval/dataset.jsonl", help="取 gold 标注")
    parser.add_argument("--multiturn", default="eval/multiturn.jsonl", help="多轮追问用例")
    parser.add_argument("--kb-id", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--modes", default="off,concat,llm,hybrid", help="要对比的策略，逗号分隔")
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default="", help="默认取 settings.deepseek_model")
    parser.add_argument("--dry-run", action="store_true", help="只跑改写，不检索（看改写质量）")
    parser.add_argument("--label", default="", help="报告文件名里的场景标签，如 chain")
    args = parser.parse_args()
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in args.modes if m not in MODES]
    if unknown:
        raise SystemExit(f"未知策略：{unknown}，可选 {MODES}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
