"""从 Qdrant 抽样片段，用 LLM 反向生成评测问题（gold = 该片段）。

三种题型：
  single    —— 读一个文本片段，写一个用户会问的问题（单跳事实）
  multi_hop —— 读两个**不同文档**的片段，写一个必须同时用到两者的比较/综合问题
  image     —— 读一条图片 caption，写一个用户看到图才会问的问题

为什么反向造题：RAG 评测最难的是标注。让 LLM 读真实片段、写自然提问，天然拿到
gold_chunk_id。缺点与对策见 docs/EVAL_PLAYBOOK.md（禁止照抄原文、问题必须唯一可解）。

用法（仓库根目录，用项目 venv）：
    .venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --mode single --limit 20
    .venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --mode multi_hop --limit 20 --append
    .venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --mode image --limit 15 --append
    .venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --mode image --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

# 封面/版权页/宣传页的 caption 有问答价值低，做图片题时排除（否则评的是一堆条形码）
_IMAGE_SKIP_HINTS = (
    "条形码", "定价", "CIP", "版权页", "封底", "书脊", "扉页",
    "宣传海报", "宣传页", "出版信息", "ISBN",
)

SINGLE_PROMPT = """你在为一个 RAG 知识库构造评测集。下面是文档中的一个片段（来自《{source}》{section}）：

<<<
{text}
>>>

请基于这个片段写一条「用户会真实问出来的问题」，并给出答案要点。要求：

1. 问题必须能**完全**由该片段回答，不要问片段里没有的信息。
2. **不要照抄原文短语**：把关键术语替换成同义表达或上位词（原文连续超过 8 个字的片段不要出现在问题里）。
3. 问题要具体，避免"这一章讲了什么"这类过泛的问法。
4. query_type 取值：single_hop（单点事实）/ exact_term（依赖专有名词、数字、编号）/ paraphrase（完全口语化改写）。
5. answer_points：2-3 条简短要点，用于判断回答是否正确。

只输出 JSON：
{{"question": "...", "query_type": "single_hop|exact_term|paraphrase", "answer_points": ["..."], "notes": "为什么只能由该片段回答"}}
"""

IMAGE_PROMPT = """你在为一个图文知识库构造评测集。下面是一张插图/图表的中文描述（来自《{source}》{section}）：

<<<
{text}
>>>

请写一条「用户看了这张图会问」的问题，并给出答案要点。要求：

1. 问题要指向图里呈现的信息（趋势、结构、对比、标注等），而不是泛泛问主题。
2. 不要出现"这张图"以外的指代，也不要说"图片描述里写了什么"——要像读者自然发问。
3. 不要照抄描述里的长句（连续超过 8 个字不要复用）。
4. answer_points：2 条要点。

只输出 JSON：
{{"question": "...", "query_type": "image", "answer_points": ["..."], "notes": "图中哪个信息能回答"}}
"""

MULTI_HOP_PROMPT = """你在为一个 RAG 知识库构造「多跳」评测题。下面是**两本不同书**中的片段：

【片段 A｜《{source_a}》{section_a}】
<<<
{text_a}
>>>

【片段 B｜《{source_b}》{section_b}】
<<<
{text_b}
>>>

请写一条必须**同时**用到 A 和 B 才能回答的问题（比较异同、建立联系、或综合两者观点）。要求：

1. 只用其中一段答不出来，必须两段都用上。
2. 不要照抄原文长句（连续超过 8 个字不要复用）。
3. 问题要自然，像读者读了两本书后的疑问。
4. answer_points：3 条要点，其中至少 1 条分别来自 A 和 B。

只输出 JSON：
{{"question": "...", "query_type": "multi_hop", "answer_points": ["..."], "notes": "A 提供什么、B 提供什么"}}
"""


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def parse_kv(text: str) -> dict[str, Any]:
    match = _JSON_BLOCK.search(text or "")
    if not match:
        raise ValueError("模型输出里没有 JSON")
    return json.loads(match.group(0))


def sample_chunks(
    collection: str,
    per_doc: int,
    max_docs: int,
    host: str,
    port: int,
    seed: int,
    chunk_type: str = "text",
    min_len: int = 200,
) -> list[dict[str, Any]]:
    """按 doc_id 分层抽样：每本书取若干片段，避免问题全来自同一本书"""
    client = QdrantClient(host=host, port=port, timeout=30)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    offset = None
    scanned = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            text = (payload.get("text") or "").strip()
            if payload.get("chunk_type") != chunk_type or len(text) < min_len:
                continue
            doc_id = payload.get("doc_id") or "unknown"
            by_doc.setdefault(doc_id, []).append(
                {
                    "chunk_id": str(point.id),
                    "doc_id": doc_id,
                    "source_doc": payload.get("source_doc"),
                    "page_num": payload.get("page_num"),
                    "section_title": payload.get("section_title"),
                    "image_path": payload.get("image_path"),
                    "text": text,
                }
            )
        scanned += len(points)
        if offset is None or scanned > 40000:
            break

    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for doc_id in docs[:max_docs]:
        chunks = by_doc[doc_id]
        chunks.sort(key=lambda c: (c["section_title"] is None, -len(c["text"])))
        picked.extend(chunks[:per_doc])
    rng.shuffle(picked)
    return picked


def sample_pairs(chunks: list[dict[str, Any]], count: int, seed: int) -> list[tuple[dict, dict]]:
    """抽跨文档片段对（同一篇文档内的两段不叫多跳）"""
    rng = random.Random(seed)
    pairs: list[tuple[dict, dict]] = []
    seen: set[frozenset[str]] = set()
    pool = list(chunks)
    rng.shuffle(pool)
    for i, first in enumerate(pool):
        if len(pairs) >= count:
            break
        for second in pool[i + 1 :]:
            if second["doc_id"] == first["doc_id"]:
                continue
            key = frozenset((first["chunk_id"], second["chunk_id"]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((first, second))
            break
    return pairs


async def call_llm(
    client: httpx.AsyncClient, base_url: str, api_key: str, model: str, prompt: str
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # 稍高，减少问题与原文用词重合
        # 推理模型的思考 token 也计入 max_tokens，太小会「只有思考、正文为空」
        "max_tokens": 2000,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    content = choice["message"]["content"]
    if not (content or "").strip():
        raise ValueError(f"正文为空（finish_reason={choice.get('finish_reason')}）")
    return parse_kv(content)


async def gen_single(client, base_url, api_key, model, chunk) -> dict[str, Any] | None:
    section = f"「{chunk['section_title']}」" if chunk.get("section_title") else ""
    prompt = SINGLE_PROMPT.format(
        source=chunk.get("source_doc") or "未知文档", section=section, text=chunk["text"][:1600]
    )
    data = await call_llm(client, base_url, api_key, model, prompt)
    return {
        "question": (data.get("question") or "").strip(),
        "type": data.get("query_type") or "single_hop",
        "answer_points": data.get("answer_points") or [],
        "notes": data.get("notes"),
    }


async def gen_image(client, base_url, api_key, model, chunk) -> dict[str, Any] | None:
    section = f"「{chunk['section_title']}」" if chunk.get("section_title") else ""
    prompt = IMAGE_PROMPT.format(
        source=chunk.get("source_doc") or "未知文档", section=section, text=chunk["text"][:1200]
    )
    data = await call_llm(client, base_url, api_key, model, prompt)
    return {
        "question": (data.get("question") or "").strip(),
        "type": "image",
        "answer_points": data.get("answer_points") or [],
        "notes": data.get("notes"),
    }


async def gen_multi_hop(client, base_url, api_key, model, pair) -> dict[str, Any] | None:
    first, second = pair
    prompt = MULTI_HOP_PROMPT.format(
        source_a=first.get("source_doc") or "未知",
        section_a=f"「{first['section_title']}」" if first.get("section_title") else "",
        text_a=first["text"][:1200],
        source_b=second.get("source_doc") or "未知",
        section_b=f"「{second['section_title']}」" if second.get("section_title") else "",
        text_b=second["text"][:1200],
    )
    data = await call_llm(client, base_url, api_key, model, prompt)
    return {
        "question": (data.get("question") or "").strip(),
        "type": "multi_hop",
        "answer_points": data.get("answer_points") or [],
        "notes": data.get("notes"),
    }


async def main_async(args: argparse.Namespace) -> None:
    env = load_env()
    host = args.qdrant_host or env.get("QDRANT_HOST", "localhost")
    port = int(args.qdrant_port or env.get("QDRANT_PORT", "6333"))

    print(f"[{args.mode}] 抽样：collection={args.collection} host={host}:{port}")
    if args.mode == "image":
        chunks = sample_chunks(
            args.collection, args.per_doc, args.max_docs, host, port, args.seed,
            chunk_type="image_caption", min_len=60,
        )
        before = len(chunks)
        chunks = [c for c in chunks if not any(h in c["text"] for h in _IMAGE_SKIP_HINTS)]
        print(f"图片片段：{before} → {len(chunks)}（已排除封面/版权页/宣传页）")
    else:
        chunks = sample_chunks(args.collection, args.per_doc, args.max_docs, host, port, args.seed)

    print(f"候选片段：{len(chunks)}（{len({c['doc_id'] for c in chunks})} 篇文档）")
    if not chunks:
        raise SystemExit("没有抽到片段：检查 collection 名与 chunk_type")

    if args.dry_run:
        for chunk in chunks[: args.limit]:
            print(f"- {chunk['chunk_id'][:8]} {chunk['source_doc']}: {chunk['text'][:90]}")
        return

    api_key = env.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY（写入 .env 或设置环境变量）")
    base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = args.model or env.get("DEEPSEEK_MODEL", "deepseek-chat")

    rows: list[dict[str, Any]] = []
    prefix = {"single": "s", "image": "i", "multi_hop": "m"}[args.mode]

    async with httpx.AsyncClient(timeout=180) as client:
        if args.mode == "multi_hop":
            pairs = sample_pairs(chunks, args.limit, args.seed)
            print(f"跨文档组合：{len(pairs)} 组")
            for index, pair in enumerate(pairs, 1):
                first, second = pair
                try:
                    result = await gen_multi_hop(client, base_url, api_key, model, pair)
                except Exception as exc:
                    print(f"  [warn] 第 {index} 组失败：{type(exc).__name__}: {exc}")
                    continue
                if not result["question"]:
                    continue
                rows.append(
                    {
                        "id": f"{prefix}{len(rows) + 1:03d}",
                        "kb_id": args.kb_id,
                        "collection": args.collection,
                        "question": result["question"],
                        "type": "multi_hop",
                        "gold_chunk_ids": [first["chunk_id"], second["chunk_id"]],
                        "gold_doc_ids": [first["doc_id"], second["doc_id"]],
                        "answer_points": result["answer_points"],
                        "source_doc": f"{first.get('source_doc')} + {second.get('source_doc')}",
                        "gold_text_preview": (first["text"][:150] + " … " + second["text"][:150]),
                        "generated_by": f"{model}@reverse-question",
                        "notes": result.get("notes"),
                    }
                )
                print(f"[{index}/{len(pairs)}] {result['question'][:60]}")
        else:
            generator = gen_image if args.mode == "image" else gen_single
            for index, chunk in enumerate(chunks[: args.limit], 1):
                try:
                    result = await generator(client, base_url, api_key, model, chunk)
                except Exception as exc:
                    print(f"  [warn] 生成失败 {chunk['chunk_id'][:8]}: {type(exc).__name__}: {exc}")
                    continue
                if not result["question"]:
                    continue
                if args.types and result["type"] not in args.types:
                    continue
                rows.append(
                    {
                        "id": f"{prefix}{len(rows) + 1:03d}",
                        "kb_id": args.kb_id,
                        "collection": args.collection,
                        "question": result["question"],
                        "type": result["type"],
                        "gold_chunk_ids": [chunk["chunk_id"]],
                        "gold_doc_ids": [chunk["doc_id"]],
                        "answer_points": result["answer_points"],
                        "source_doc": chunk["source_doc"],
                        "section_title": chunk.get("section_title"),
                        "gold_text_preview": chunk["text"][:300],
                        "generated_by": f"{model}@reverse-question",
                        "notes": result.get("notes"),
                    }
                )
                print(f"[{index}/{args.limit}] {result['type']:10} {result['question'][:60]}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 追加时重排 id，避免与已有数据冲突
    existing_ids: set[str] = set()
    if args.append and out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line).get("id", ""))
        for row in rows:
            base = row["id"]
            suffix = 1
            while row["id"] in existing_ids:
                row["id"] = f"{base}-{suffix}"
                suffix += 1
            existing_ids.add(row["id"])

    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n已{'追加' if args.append else '写入'} {out_path}（{len(rows)} 条）")

    dist: dict[str, int] = {}
    for row in rows:
        dist[row["type"]] = dist.get(row["type"], 0) + 1
    print("本次类型分布：", dist)


def main() -> None:
    parser = argparse.ArgumentParser(description="从知识库反向生成 RAG 评测集")
    parser.add_argument("--collection", required=True, help="Qdrant 集合名")
    parser.add_argument("--kb-id", default="")
    parser.add_argument("--mode", choices=["single", "multi_hop", "image"], default="single")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--per-doc", type=int, default=2)
    parser.add_argument("--max-docs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--types", default="", help="single 模式下过滤题型，逗号分隔")
    parser.add_argument("--model", default="")
    parser.add_argument("--qdrant-host", default="")
    parser.add_argument("--qdrant-port", default="")
    parser.add_argument("--out", default="eval/dataset.jsonl")
    parser.add_argument("--append", action="store_true", help="追加到现有数据集")
    parser.add_argument("--dry-run", action="store_true", help="只抽样打印，不调 LLM")
    args = parser.parse_args()
    args.types = [t.strip() for t in args.types.split(",") if t.strip()]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
