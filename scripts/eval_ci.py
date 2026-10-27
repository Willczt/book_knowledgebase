"""知识库更新后的一键回归：gold 体检 → 跑检索(on/off) → 对比基线 → 出报告 → 退出码。

为什么需要它：知识库一更新（新增文档/重摄入），评测集里的 `gold_chunk_id` 可能整体失效
（Qdrant point id 是 uuid4，重新摄入就会变）。如果直接跑，指标会莫名暴跌，而真实原因是
**评测数据过期**，不是系统变差。所以第一步必须先体检。

用法：
    # 日常回归（与 eval/baseline.json 对比，掉点即失败）
    .venv/bin/python scripts/eval_ci.py --kb-id <kb_id> --collection kb_xxx

    # 更新基线（确认这次改动是"新的标准"后执行，会写 eval/baseline.json）
    .venv/bin/python scripts/eval_ci.py --kb-id <kb_id> --collection kb_xxx --update-baseline

    # 只体检不跑分
    .venv/bin/python scripts/eval_ci.py --kb-id <kb_id> --collection kb_xxx --check-only

退出码：0 通过 / 1 指标退化 / 2 gold 失效 / 3 服务不可用
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_rag import load_dataset, run_retrieval  # noqa: E402


def load_env() -> dict[str, str]:
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


def check_gold(cases, collection: str, host: str, port: int) -> dict[str, Any]:
    """体检 gold 片段是否还在库里（KB 重摄入会让 point id 全部变化）"""
    ids = sorted({cid for case in cases for cid in case.gold_chunk_ids})
    if not ids:
        return {"total": 0, "missing": 0, "stale_case_ids": [], "ok": True}

    client = QdrantClient(host=host, port=port, timeout=30)
    found: set[str] = set()
    for start in range(0, len(ids), 256):
        batch = ids[start : start + 256]
        points = client.retrieve(collection_name=collection, ids=batch, with_payload=False)
        found |= {str(point.id) for point in points}

    missing_ids = [cid for cid in ids if cid not in found]
    missing_set = set(missing_ids)
    stale_cases = [
        case.id
        for case in cases
        if case.gold_chunk_ids and all(cid in missing_set for cid in case.gold_chunk_ids)
    ]
    return {
        "total": len(ids),
        "missing": len(missing_ids),
        "stale_case_ids": stale_cases,
        "ok": not missing_ids,
    }


def summarize_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("hit@1", "recall@10", "mrr@10", "ndcg@10")
    return {k: round(current.get(k, 0) - baseline.get(k, 0), 4) for k in keys if k in baseline}


def render(report: dict[str, Any]) -> str:
    lines = ["# RAG 回归报告", ""]
    meta = report["meta"]
    lines += [
        f"- 时间：{meta['time']}",
        f"- 知识库：`{meta['kb_id']}`（collection `{meta['collection']}`）",
        f"- 数据集：{meta['dataset']} · {meta['n_cases']} 条 · 指纹 `{meta['dataset_sha']}`",
        "",
        "## 0. 评测数据体检",
        "",
    ]
    gold = report["gold"]
    if gold["ok"]:
        lines.append(f"- gold 片段全部存在（{gold['total']} 个）")
    else:
        lines.append(
            f"- ⚠️ {gold['missing']}/{gold['total']} 个 gold 片段在库里找不到，"
            f"涉及 {len(gold['stale_case_ids'])} 道题"
        )
        lines.append(
            f"- 受影响题目：`{', '.join(gold['stale_case_ids'][:10])}`"
            + ("…" if len(gold["stale_case_ids"]) > 10 else "")
        )
        lines.append(
            "- 处理办法：确认是否重摄入过对应文档。若是，需用 `gen_eval_set.py` 重造这几本书的题，"
            "或把 `gold_chunk_ids` 手工更新为新 id（报告 JSON 里 `per_case` 可查）"
        )

    for key, title in (("off", "关闭精排"), ("on", "开启精排")):
        section = report["retrieval"][key]
        overall = section["overall"]
        lines += [
            "",
            f"## 1.{'1' if key == 'off' else '2'} 检索指标（{title}）",
            "",
            "| 切片 | n | Hit@1 | R@5 | R@10 | R@30 | MRR@10 | NDCG@10 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, stats in [("**全部**", overall)] + list(section["by_type"].items()):
            lines.append(
                f"| {name} | {stats['n']} | {stats['hit@1']:.3f} | {stats['recall@5']:.3f} | "
                f"{stats['recall@10']:.3f} | {stats['recall@30']:.3f} | {stats['mrr@10']:.3f} | "
                f"{stats['ndcg@10']:.3f} |"
            )
        latency = section["latency"]
        lines += [
            "",
            f"- 检索延迟：P50 {latency['p50_ms']} ms / P95 {latency['p95_ms']} ms",
            f"- 失败请求：{section['errors']}",
        ]

    if report.get("delta"):
        lines += ["", "## 2. 与基线对比", "", "| 指标 | 基线 | 本次 | 变化 |", "|---|---|---|---|"]
        for key, value in report["delta"].items():
            base = report["baseline"][key]
            mark = "🟢" if value > 0.01 else ("🔴" if value < -0.01 else "⚪")
            lines.append(f"| {key} | {base:.3f} | {base + value:.3f} | {value:+.3f} {mark} |")
        if report["regressions"]:
            lines += ["", f"> ❌ 退化指标：{', '.join(report['regressions'])}"]
        else:
            lines += ["", "> ✅ 无超过阈值的退化"]

    lines += ["", "## 3. 结论", ""]
    lines.append(report["verdict"])
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    env = load_env()
    dataset_path = ROOT / args.dataset
    cases = load_dataset(dataset_path)
    with_gold = [case for case in cases if case.gold_chunk_ids]
    print(f"数据集：{len(cases)} 条（{len(with_gold)} 条有 gold）")

    host = args.qdrant_host or env.get("QDRANT_HOST", "localhost")
    port = int(args.qdrant_port or env.get("QDRANT_PORT", "6333"))

    print("\n[1/3] gold 片段体检")
    gold = check_gold(cases, args.collection, host, port)
    print(f"  gold {gold['total']} 个，缺失 {gold['missing']}，受影响题目 {len(gold['stale_case_ids'])}")

    if args.check_only:
        print("\n仅体检模式，退出")
        return 0 if gold["ok"] else 2

    # 服务可用性
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{args.base_url}/api/v1/health")
            resp.raise_for_status()
    except Exception as exc:
        print(f"  ❌ 后端不可用：{type(exc).__name__}: {exc}")
        return 3

    params_off = {"top_k": 30, "threshold": 0.0, "min_rerank_score": 0.0, "use_reranker": False}
    params_on = {**params_off, "use_reranker": True}

    print("\n[2/3] 检索评测：关闭精排")
    off = await run_retrieval(with_gold, args.base_url, args.kb_id, params_off)
    print("\n[3/3] 检索评测：开启精排")
    on = await run_retrieval(with_gold, args.base_url, args.kb_id, params_on)

    report: dict[str, Any] = {
        "meta": {
            "time": datetime.now().isoformat(timespec="seconds"),
            "kb_id": args.kb_id,
            "collection": args.collection,
            "dataset": args.dataset,
            "dataset_sha": hashlib.sha1(dataset_path.read_bytes()).hexdigest()[:8],
            "n_cases": len(with_gold),
        },
        "gold": gold,
        "retrieval": {"off": off, "on": on},
        "delta": None,
        "regressions": [],
    }

    baseline_path = ROOT / args.baseline
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("dataset_sha") and baseline["dataset_sha"] != report["meta"]["dataset_sha"]:
            print("  [warn] 基线数据集指纹与当前不一致，跳过对比")
        else:
            report["baseline"] = baseline["on"] if args.compare == "on" else baseline["off"]
            report["delta"] = summarize_delta(off["overall"] if args.compare == "off" else on["overall"], report["baseline"])
            report["regressions"] = [
                key for key, value in report["delta"].items() if value < -args.fail_on_drop / 100
            ]

    # 结论
    on_head = on["overall"]["hit@1"]
    off_head = off["overall"]["hit@1"]
    if not gold["ok"]:
        verdict = (
            f"评测数据有 {gold['missing']} 个 gold 失效，指标仅供参考。"
            "先修复评测集（重造对应文档的题），再判断系统是否退化。"
        )
    elif report["regressions"]:
        verdict = f"检测到退化：{', '.join(report['regressions'])} 超过 {args.fail_on_drop} 个百分点，需要排查。"
    else:
        verdict = (
            f"通过。开精排 Hit@1 {on_head:.3f}，关闭 {off_head:.3f}，"
            f"差距 {on_head - off_head:+.3f}，与预期一致。"
        )
    report["verdict"] = verdict

    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ci_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    markdown = render(report)
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + markdown)
    print(f"\n报告：{out_dir / f'{stem}.md'}")

    if args.update_baseline:
        baseline = {
            "updated_at": report["meta"]["time"],
            "dataset": args.dataset,
            "dataset_sha": report["meta"]["dataset_sha"],
            "n_cases": report["meta"]["n_cases"],
            "off": off["overall"],
            "on": on["overall"],
        }
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"基线已更新：{baseline_path}")

    if not gold["ok"]:
        return 2
    if report["regressions"]:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库更新后的 RAG 回归")
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--collection", required=True, help="Qdrant 集合名，如 kb_xxx")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--baseline", default="eval/baseline.json")
    parser.add_argument("--compare", choices=["off", "on"], default="on", help="用哪组指标和基线比")
    parser.add_argument("--fail-on-drop", type=float, default=5.0, dest="fail_on_drop")
    parser.add_argument("--qdrant-host", default="")
    parser.add_argument("--qdrant-port", default="")
    parser.add_argument("--check-only", action="store_true", help="只做 gold 体检")
    parser.add_argument("--update-baseline", action="store_true", help="把本次指标写成新基线")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
