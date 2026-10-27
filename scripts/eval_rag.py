"""RAG 效果评测：检索指标 + 参数扫描 + 生成质量（可选）。

分两层，互不依赖：
  1) retrieval —— 只调 /search，不花 LLM token。可高频跑、可做参数网格。
     指标：Recall@K / MRR@K / NDCG@K / Hit@1，按题型切片。
  2) generation —— 走 /sessions/{id}/chat/stream，收集回答。
     指标：引用覆盖率、拒答正确率、TTFT、总时长；加 --judge 时用 LLM 评 faithfulness/correctness。

用法（仓库根目录）：
    # 基线
    .venv/bin/python scripts/eval_rag.py --dataset eval/dataset.jsonl \
        --kb-id xxx --base-url http://127.0.0.1:8001 --stage retrieval --label baseline

    # 参数扫描（单维度）
    .venv/bin/python scripts/eval_rag.py ... --sweep "top_k=3,5,8,15"
    .venv/bin/python scripts/eval_rag.py ... --sweep "threshold=0,0.2,0.35"
    .venv/bin/python scripts/eval_rag.py ... --sweep "rerank=on,off"

    # 生成质量（每题新建会话，避免历史污染）
    .venv/bin/python scripts/eval_rag.py ... --stage generation --judge \
        --llm-provider deepseek --llm-model deepseek-flash

输出：eval/reports/<label>_<ts>.md 与同名 .json
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import httpx

ROOT = Path(__file__).resolve().parent.parent
_SENTENCE_SPLIT = re.compile(r"[。！？；!?;\n]+")
_REFUSAL_HINTS = ("没有找到", "未提及", "无法回答", "资料中没有", "参考资料中没有", "不清楚", "无法确定")


# ============================================================
# 数据集
# ============================================================


@dataclass
class Case:
    id: str
    question: str
    type: str
    gold_chunk_ids: list[str]
    answer_points: list[str] = field(default_factory=list)
    source_doc: str | None = None
    expected_refusal: bool = False


def load_dataset(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                Case(
                    id=row.get("id") or f"q{len(cases) + 1:03d}",
                    question=row["question"],
                    type=row.get("type") or "single_hop",
                    gold_chunk_ids=row.get("gold_chunk_ids") or [],
                    answer_points=row.get("answer_points") or [],
                    source_doc=row.get("source_doc"),
                    expected_refusal=bool(row.get("expected_refusal")) or row.get("type") == "no_answer",
                )
            )
    return cases


# ============================================================
# 指标（口径与后端 core/observability/trace.py 保持一致）
# ============================================================


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(ranked[:k]) & set(gold))
    return hit / len(gold)


def precision_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return len(set(top) & set(gold)) / len(top)


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    for index, chunk_id in enumerate(ranked[:k], 1):
        if chunk_id in gold_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in gold_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return dcg / ideal if ideal else 0.0


def hit_at_1(ranked: list[str], gold: list[str]) -> float:
    return 1.0 if ranked and ranked[0] in set(gold) else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _pctl(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * ratio
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 1)


# ============================================================
# 检索评测
# ============================================================


async def search_one(
    client: httpx.AsyncClient,
    base_url: str,
    kb_id: str,
    question: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "query": question,
        "top_k": params.get("top_k", 30),
        "score_threshold": params.get("threshold", 0.0),
        "min_rerank_score": params.get("min_rerank_score", 0.0),
        "use_reranker": params.get("use_reranker"),
    }
    if params.get("rerank_candidates"):
        body["rerank_candidates"] = params["rerank_candidates"]
    started = time.perf_counter()
    resp = await client.post(f"{base_url}/api/v1/knowledge-bases/{kb_id}/search", json=body)
    latency = (time.perf_counter() - started) * 1000
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    return {
        "ranked": [item["chunk_id"] for item in results],
        "texts": [(item["chunk_id"], item.get("text") or "", item.get("source_doc")) for item in results],
        "scores": [item.get("score") for item in results],
        "latency_ms": round(latency, 1),
    }


async def run_retrieval(
    cases: list[Case],
    base_url: str,
    kb_id: str,
    params: dict[str, Any],
    show_misses: int = 5,
) -> dict[str, Any]:
    # 无答案题没有 gold 片段，检索指标无意义（它们只参与生成侧的拒答评估）
    skipped = len([c for c in cases if not c.gold_chunk_ids])
    if skipped:
        print(f"  跳过 {skipped} 条无 gold 的题（仅用于生成评测）")
    cases = [c for c in cases if c.gold_chunk_ids]

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=180) as client:
        for index, case in enumerate(cases, 1):
            try:
                result = await search_one(client, base_url, kb_id, case.question, params)
            except Exception as exc:
                print(f"  [{index}/{len(cases)}] 失败 {case.id}: {type(exc).__name__}: {exc}")
                rows.append({"case": case, "error": str(exc)})
                continue
            latencies.append(result["latency_ms"])
            ranked = result["ranked"]
            rows.append(
                {
                    "case": case,
                    "ranked": ranked,
                    "latency_ms": result["latency_ms"],
                    "recall_1": recall_at_k(ranked, case.gold_chunk_ids, 1),
                    "recall_3": recall_at_k(ranked, case.gold_chunk_ids, 3),
                    "recall_5": recall_at_k(ranked, case.gold_chunk_ids, 5),
                    "recall_8": recall_at_k(ranked, case.gold_chunk_ids, 8),
                    "recall_10": recall_at_k(ranked, case.gold_chunk_ids, 10),
                    "recall_30": recall_at_k(ranked, case.gold_chunk_ids, 30),
                    "mrr_10": mrr_at_k(ranked, case.gold_chunk_ids, 10),
                    "ndcg_10": ndcg_at_k(ranked, case.gold_chunk_ids, 10),
                    "hit_1": hit_at_1(ranked, case.gold_chunk_ids),
                    "top1_text": (result["texts"][0][1][:200] if result["texts"] else ""),
                    "top1_doc": (result["texts"][0][2] if result["texts"] else None),
                }
            )
            flag = "OK  " if rows[-1].get("recall_10") else "MISS"
            rank = next(
                (i for i, cid in enumerate(ranked, 1) if cid in set(case.gold_chunk_ids)), None
            )
            print(
                f"  [{index}/{len(cases)}] {flag} rank={rank or '-':>3} "
                f"R@10={rows[-1].get('recall_10', 0):.2f} {case.question[:40]}"
            )

    scored = [row for row in rows if "error" not in row]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_type.setdefault(row["case"].type, []).append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "hit@1": _mean(r["hit_1"] for r in group),
            "recall@1": _mean(r["recall_1"] for r in group),
            "recall@3": _mean(r["recall_3"] for r in group),
            "recall@5": _mean(r["recall_5"] for r in group),
            "recall@8": _mean(r["recall_8"] for r in group),
            "recall@10": _mean(r["recall_10"] for r in group),
            "recall@30": _mean(r["recall_30"] for r in group),
            "mrr@10": _mean(r["mrr_10"] for r in group),
            "ndcg@10": _mean(r["ndcg_10"] for r in group),
        }

    misses = [r for r in scored if r["recall_10"] == 0][:show_misses]
    return {
        "stage": "retrieval",
        "params": params,
        "n": len(scored),
        "errors": len(rows) - len(scored),
        "overall": summarize(scored),
        "by_type": {name: summarize(group) for name, group in sorted(by_type.items())},
        "latency": {
            "p50_ms": _pctl(latencies, 0.5),
            "p95_ms": _pctl(latencies, 0.95),
            "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        },
        "misses": [
            {
                "id": r["case"].id,
                "question": r["case"].question,
                "gold_doc": r["case"].source_doc,
                "gold_preview": (r["case"].gold_chunk_ids and "") or "",
                "top1_doc": r.get("top1_doc"),
                "top1_text": r.get("top1_text"),
            }
            for r in misses
        ],
        "rows": [
            {"id": r["case"].id, "type": r["case"].type, "hit_1": r["hit_1"], "recall_10": r["recall_10"], "ranked": r["ranked"][:10]}
            for r in scored
        ],
    }


# ============================================================
# 生成评测（含可选 LLM judge）
# ============================================================


async def create_session(client: httpx.AsyncClient, base_url: str, kb_id: str, provider: str, model: str) -> str:
    resp = await client.post(
        f"{base_url}/api/v1/sessions",
        json={"kb_id": kb_id, "llm_provider": provider, "llm_model_id": model, "title": "eval"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def chat_once(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    question: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "content": question,
        "retrieval_config": {
            "top_k": params.get("top_k", 8),
            "score_threshold": params.get("threshold", 0.0),
            "min_rerank_score": params.get("min_rerank_score", 0.0),
            "use_reranker": params.get("use_reranker"),
        },
    }
    tokens: list[str] = []
    sources: list[dict[str, Any]] = []
    error: str | None = None
    ttft: float | None = None
    started = time.perf_counter()

    async with client.stream(
        "POST", f"{base_url}/api/v1/sessions/{session_id}/chat/stream", json=body
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "token" in event:
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                tokens.append(event["token"])
            elif event.get("done"):
                sources = event.get("sources") or []
            elif "error" in event:
                error = event["error"]

    return {
        "answer": "".join(tokens),
        "sources": sources,
        "error": error,
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def citation_coverage(answer: str, sources: list[dict[str, Any]]) -> float:
    """回答里有几句能对上来源片段（与后端 _match_citations 同口径，弱指标）"""
    if not sources:
        return 0.0
    normalized = re.sub(r"\s+", "", answer or "")
    cited = 0
    for source in sources:
        text = source.get("text") or ""
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = re.sub(r"\s+", "", sentence)
            if len(sentence) >= 12 and sentence[:20] in normalized:
                cited += 1
                break
    return round(cited / len(sources), 4)


def looks_refused(answer: str) -> bool:
    text = (answer or "")[:400]
    return any(hint in text for hint in _REFUSAL_HINTS)


JUDGE_PROMPT = """你是严格的 RAG 评测员。根据【问题】【参考资料】【回答】打分，只输出 JSON。

问题：{question}

参考资料（检索到的片段，可能不完整）：
{context}

回答：
{answer}

评分维度（1-5 整数，5 最好）：
- faithfulness：回答中的事实性断言是否都能在上面的参考资料里找到依据？有编造即扣分。
- correctness：是否覆盖了这些要点：{points}
- relevance：是否直接回答了问题（答非所问给 1-2 分）。

只输出：{{"faithfulness": 1-5, "correctness": 1-5, "relevance": 1-5, "reason": "不超过 40 字"}}
"""


async def judge_answer(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    points: list[str],
) -> dict[str, Any]:
    context = "\n\n".join(
        f"[{i}] {(s.get('text') or '')[:600]}" for i, s in enumerate(sources[:8], 1)
    ) or "（无参考资料）"
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer[:3000],
        points="；".join(points) or "（未提供要点，按问题本身判断）",
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    return json.loads(match.group(0)) if match else {}


async def run_generation(
    cases: list[Case],
    base_url: str,
    kb_id: str,
    params: dict[str, Any],
    provider: str,
    model: str,
    judge: bool,
    judge_key: str,
    judge_base: str,
    judge_model: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=300) as client:
        for index, case in enumerate(cases, 1):
            try:
                # 每题独立会话：复用会带入历史，后续题的上下文被污染，指标不可比
                session_id = await create_session(client, base_url, kb_id, provider, model)
                result = await chat_once(client, base_url, session_id, case.question, params)
            except Exception as exc:
                print(f"  [{index}/{len(cases)}] 失败 {case.id}: {type(exc).__name__}: {exc}")
                continue
            coverage = citation_coverage(result["answer"], result["sources"])
            refused = looks_refused(result["answer"])
            scores: dict[str, Any] = {}
            if judge and result["answer"] and judge_key:
                try:
                    scores = await judge_answer(
                        client, judge_base, judge_key, judge_model,
                        case.question, result["answer"], result["sources"], case.answer_points,
                    )
                except Exception as exc:
                    print(f"     judge 失败 {case.id}: {type(exc).__name__}")
            rows.append(
                {
                    "id": case.id,
                    "type": case.type,
                    "question": case.question,
                    "answer": result["answer"],
                    "error": result["error"],
                    "coverage": coverage,
                    "refused": refused,
                    "expected_refusal": case.expected_refusal,
                    "ttft_ms": result["ttft_ms"],
                    "total_ms": result["total_ms"],
                    "scores": scores,
                }
            )
            print(
                f"  [{index}/{len(cases)}] cover={coverage:.2f} "
                f"ttft={result['ttft_ms'] or 0:.0f}ms "
                f"judge={scores.get('faithfulness', '-')}/{scores.get('correctness', '-')}"
            )

    answerable = [r for r in rows if not r["expected_refusal"]]
    unanswerable = [r for r in rows if r["expected_refusal"]]
    judged = [r for r in rows if r["scores"]]

    return {
        "stage": "generation",
        "params": params,
        "n": len(rows),
        "citation_coverage": _mean(r["coverage"] for r in rows if r["answer"]),
        "refusal_rate_on_unanswerable": _mean(1.0 if r["refused"] else 0.0 for r in unanswerable),
        "false_refusal_rate": _mean(1.0 if r["refused"] else 0.0 for r in answerable),
        "ttft": {
            "p50_ms": _pctl([r["ttft_ms"] for r in rows if r["ttft_ms"]], 0.5),
            "p95_ms": _pctl([r["ttft_ms"] for r in rows if r["ttft_ms"]], 0.95),
        },
        "total": {
            "p50_ms": _pctl([r["total_ms"] for r in rows], 0.5),
            "p95_ms": _pctl([r["total_ms"] for r in rows], 0.95),
        },
        "judge": {
            "n": len(judged),
            "faithfulness": _mean(r["scores"].get("faithfulness", 0) for r in judged),
            "correctness": _mean(r["scores"].get("correctness", 0) for r in judged),
            "relevance": _mean(r["scores"].get("relevance", 0) for r in judged),
        },
        "rows": rows,
    }


# ============================================================
# 报告
# ============================================================


def render_markdown(title: str, reports: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- 时间：{meta['time']}")
    lines.append(f"- 数据集：{meta['dataset']}（{meta['n_cases']} 条）")
    lines.append(f"- 后端：{meta['base_url']}")
    lines.append("")

    for report in reports:
        if report["stage"] == "retrieval":
            params = report["params"]
            lines.append(
                f"## 检索 · `{' '.join(f'{k}={v}' for k, v in params.items())}`"
            )
            lines.append("")
            lines.append(f"- 样本：{report['n']}（失败 {report['errors']}）")
            lines.append(
                f"- 延迟：P50 {report['latency']['p50_ms']} ms / P95 {report['latency']['p95_ms']} ms"
            )
            lines.append("")
            lines.append(
                "| 切片 | n | Hit@1 | R@1 | R@3 | R@5 | R@8 | R@10 | R@30 | MRR@10 | NDCG@10 |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            rows = [("**全部**", report["overall"])] + list(report["by_type"].items())
            for name, stats in rows:
                lines.append(
                    f"| {name} | {stats['n']} | {stats['hit@1']:.3f} | {stats['recall@1']:.3f} | "
                    f"{stats['recall@3']:.3f} | {stats['recall@5']:.3f} | {stats['recall@8']:.3f} | "
                    f"{stats['recall@10']:.3f} | {stats['recall@30']:.3f} | "
                    f"{stats['mrr@10']:.3f} | {stats['ndcg@10']:.3f} |"
                )
            if report["misses"]:
                lines.append("")
                lines.append("**未召回（Recall@10 = 0）样例：**")
                lines.append("")
                for miss in report["misses"]:
                    lines.append(f"- `{miss['id']}` {miss['question']}")
                    lines.append(f"  - 期望文档：{miss['gold_doc']}")
                    lines.append(f"  - Top1 实际：{miss['top1_doc']} ｜ {(miss['top1_text'] or '')[:90]}")
            lines.append("")

        elif report["stage"] == "generation":
            lines.append(f"## 生成 · `{' '.join(f'{k}={v}' for k, v in report['params'].items())}`")
            lines.append("")
            lines.append(f"- 样本：{report['n']}")
            lines.append(f"- 引用覆盖率：{report['citation_coverage']}")
            lines.append(f"- 拒答率（无答案题）：{report['refusal_rate_on_unanswerable']}")
            lines.append(f"- 误拒率（有答案题）：{report['false_refusal_rate']}")
            lines.append(
                f"- TTFT：P50 {report['ttft']['p50_ms']} ms / P95 {report['ttft']['p95_ms']} ms"
            )
            lines.append(
                f"- 总耗时：P50 {report['total']['p50_ms']} ms / P95 {report['total']['p95_ms']} ms"
            )
            if report["judge"]["n"]:
                judge = report["judge"]
                lines.append(
                    f"- LLM judge（{judge['n']} 条）：faithfulness {judge['faithfulness']} / "
                    f"correctness {judge['correctness']} / relevance {judge['relevance']}"
                )
            lines.append("")
            bad = [r for r in report["rows"] if r["scores"] and r["scores"].get("faithfulness", 5) <= 3][:5]
            if bad:
                lines.append("**faithfulness ≤ 3 的样例：**")
                lines.append("")
                for row in bad:
                    lines.append(f"- `{row['id']}` {row['question']}")
                    lines.append(f"  - 判分：{row['scores']}")
                    lines.append(f"  - 回答节选：{(row['answer'] or '')[:160]}")
            lines.append("")

    return "\n".join(lines)


# ============================================================
# 入口
# ============================================================


def parse_sweep(spec: str) -> list[dict[str, Any]]:
    """`top_k=3,5;threshold=0,0.2` → 笛卡尔积的参数字典列表"""
    axes: dict[str, list[str]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        key, values = part.split("=", 1)
        axes[key.strip()] = [v.strip() for v in values.split(",") if v.strip()]
    grids = []
    keys = list(axes)
    for combo in itertools.product(*(axes[k] for k in keys)):
        params: dict[str, Any] = {}
        for key, value in zip(keys, combo):
            if key == "top_k":
                params["top_k"] = int(value)
            elif key == "threshold":
                params["threshold"] = float(value)
            elif key == "rerank":
                params["use_reranker"] = value.lower() in ("on", "true", "1")
            elif key in ("min_rerank", "min_rerank_score"):
                params["min_rerank_score"] = float(value)
            elif key == "rerank_candidates":
                params["rerank_candidates"] = int(value)
            else:
                raise SystemExit(f"未知扫描维度：{key}")
        grids.append(params)
    return grids


async def main_async(args: argparse.Namespace) -> None:
    if not args.kb_id:
        raise SystemExit("需要 --kb-id（--self-test 除外）")
    dataset_path = ROOT / args.dataset
    cases = load_dataset(dataset_path)
    if args.limit:
        cases = cases[: args.limit]
    print(f"数据集 {dataset_path}：{len(cases)} 条，题型分布：", end=" ")
    dist: dict[str, int] = {}
    for case in cases:
        dist[case.type] = dist.get(case.type, 0) + 1
    print(dist)

    base_params: dict[str, Any] = {
        "top_k": args.top_k,
        "threshold": args.threshold,
        "min_rerank_score": args.min_rerank_score,
    }
    if args.rerank is not None:
        base_params["use_reranker"] = args.rerank
    if args.rerank_candidates:
        base_params["rerank_candidates"] = args.rerank_candidates

    reports: list[dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.stage in ("retrieval", "all"):
        grids = parse_sweep(args.sweep) if args.sweep else [base_params]
        if args.sweep:
            print(f"\n参数扫描：{len(grids)} 组")
        for index, params in enumerate(grids, 1):
            merged = {**base_params, **params}
            if "use_reranker" not in merged:
                merged["use_reranker"] = None
            print(f"\n=== 第 {index}/{len(grids)} 组：{merged} ===")
            report = await run_retrieval(cases, args.base_url, args.kb_id, merged)
            reports.append(report)

    if args.stage in ("generation", "all"):
        print("\n=== 生成评测 ===")
        env = load_generation_env()
        report = await run_generation(
            cases,
            args.base_url,
            args.kb_id,
            {**base_params, "top_k": args.gen_top_k or base_params["top_k"], "use_reranker": base_params.get("use_reranker")},
            args.llm_provider,
            args.llm_model,
            args.judge,
            env.get("DEEPSEEK_API_KEY", ""),
            env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            args.judge_model,
        )
        reports.append(report)

    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_{timestamp}"
    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "n_cases": len(cases),
        "base_url": args.base_url,
    }
    markdown = render_markdown(f"RAG 评测报告 · {args.label}", reports, meta)
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "reports": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    for report in reports:
        if report["stage"] == "retrieval":
            overall = report["overall"]
            print(
                f"{report['params']} → Hit@1 {overall['hit@1']:.3f} | "
                f"R@10 {overall['recall@10']:.3f} | MRR {overall['mrr@10']:.3f} | NDCG {overall['ndcg@10']:.3f}"
            )
        else:
            print(
                f"generation → cover {report['citation_coverage']} | "
                f"judge F{report['judge']['faithfulness']} C{report['judge']['correctness']} | "
                f"TTFT p50 {report['ttft']['p50_ms']}ms"
            )
    print(f"\n报告已写入 {out_dir / f'{stem}.md'}")
    print(markdown[:2000])


def load_generation_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def self_test() -> None:
    """不依赖任何服务，验证指标口径是否正确（改脚本后先跑这个）"""
    ranked = ["a", "b", "c", "d"]
    gold = ["c"]

    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert precision_at_k(ranked, gold, 4) == 0.25
    assert mrr_at_k(ranked, gold, 10) == 1 / 3
    assert hit_at_1(ranked, gold) == 0.0
    assert hit_at_1(["c", "a"], gold) == 1.0
    # DCG = 1/log2(2+2) = 0.5，IDCG = 1/log2(2) = 1
    assert abs(ndcg_at_k(ranked, gold, 10) - 0.5) < 1e-9
    # 多个 gold：两个都在前 2
    assert recall_at_k(["c", "d", "a"], ["c", "d"], 2) == 1.0
    assert recall_at_k(["c", "a", "d"], ["c", "d"], 2) == 0.5
    assert _pctl([1, 2, 3, 4], 0.5) == 2.5
    assert _mean([0.5, 1.0]) == 0.75
    print("self-test OK：指标口径与后端 trace.ranking_metrics 一致")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 效果评测")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--kb-id", default="", help="知识库 ID（--self-test 时可不填）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--stage", choices=["retrieval", "generation", "all"], default="retrieval")
    parser.add_argument("--label", default="baseline", help="报告文件名前缀")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--top-k", type=int, default=30, help="检索返回条数（评测需要放大才能算 R@30）")
    parser.add_argument("--threshold", type=float, default=0.0, help="召回阈值（RRF 融合分）")
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=0.0,
        dest="min_rerank_score",
        help="精排阈值（rerank 分，0~1，0 表示不过滤）",
    )
    parser.add_argument("--rerank", dest="rerank", action="store_true", help="强制开启重排")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="强制关闭重排")
    parser.set_defaults(rerank=None)
    parser.add_argument("--rerank-candidates", type=int, default=0)
    parser.add_argument("--sweep", default="", help='扫描：如 "top_k=3,5,8" 或 "threshold=0,0.2;rerank=on,off"')
    parser.add_argument("--gen-top-k", type=int, default=0, help="生成评测用的检索 top_k（默认沿用 --top-k）")
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-flash")
    parser.add_argument("--judge", action="store_true", help="启用 LLM judge（消耗 DeepSeek 额度）")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--self-test", action="store_true", help="只做指标口径自检，不连服务")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
