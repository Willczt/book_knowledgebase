"""重建 KB 集合的 BM25 稀疏向量（换分词规则后必须跑一次）。

背景：旧版用 FastEmbed `Qdrant/bm25`，中文整句被当成一个 token，BM25 基本 0 命中。
现在改用 `core.retrieval.sparse`（中文单字 + 二元组），分词规则变了，
库里的稀疏向量必须按新规则重算，否则查询与索引对不上。

用法：
    cd backend
    python -m scripts.rebuild_sparse --all
    python -m scripts.rebuild_sparse --collection kb_0569a98880a549d98aaa5486a1aeda8f
    python -m scripts.rebuild_sparse --all --dry-run

只更新稀疏向量，不重算 dense 向量，所以不会调用 embedding 模型。
"""

from __future__ import annotations

import argparse
import asyncio
import time

import structlog
from qdrant_client.models import PointVectors, SparseVector

from core.retrieval.sparse import TOKENIZER_VERSION, embed_sparse
from vector_store.qdrant_store import QdrantVectorStore

logger = structlog.get_logger()

BATCH = 128


async def rebuild_collection(store: QdrantVectorStore, collection: str, dry_run: bool) -> dict:
    client = store.client
    switched = await store.ensure_sparse_idf(collection)

    info = await client.get_collection(collection)
    total_points = int(info.points_count or 0)
    logger.info(
        "rebuild_start",
        collection=collection,
        points=total_points,
        tokenizer_version=TOKENIZER_VERSION,
        idf_switched=switched,
        dry_run=dry_run,
    )

    offset = None
    scanned = updated = skipped = 0
    started = time.perf_counter()

    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=BATCH,
            offset=offset,
            with_payload=["text"],
            with_vectors=False,
        )
        if not points:
            break

        texts = [str((point.payload or {}).get("text") or "") for point in points]
        vectors = embed_sparse(texts)

        payload: list[PointVectors] = []
        for point, (indices, values) in zip(points, vectors, strict=True):
            scanned += 1
            if not indices:
                skipped += 1
                continue
            payload.append(
                PointVectors(
                    id=point.id,
                    vector={"sparse": SparseVector(indices=indices, values=values)},
                )
            )

        if payload and not dry_run:
            await client.update_vectors(collection_name=collection, points=payload)
        updated += len(payload)

        if total_points:
            logger.info(
                "rebuild_progress",
                collection=collection,
                percent=round(min(100.0, scanned / total_points * 100), 1),
                scanned=scanned,
                updated=updated,
            )
        if offset is None:
            break

    elapsed = round(time.perf_counter() - started, 2)
    result = {
        "collection": collection,
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "idf_switched": switched,
        "elapsed_s": elapsed,
    }
    logger.info("rebuild_done", **result)
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="重建 BM25 稀疏向量")
    parser.add_argument("--collection", action="append", default=[], help="指定集合名，可重复")
    parser.add_argument("--all", action="store_true", help="处理所有 kb_ 前缀集合")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    args = parser.parse_args()

    if not args.collection and not args.all:
        parser.error("需要 --collection 或 --all")

    store = QdrantVectorStore()
    try:
        collections = list(args.collection)
        if args.all:
            existing = await store.client.get_collections()
            collections += [
                item.name for item in existing.collections if item.name.startswith("kb_")
            ]
        targets = sorted(set(collections))
        if not targets:
            logger.warning("no_collection_found")
            return

        for collection in targets:
            await rebuild_collection(store, collection, dry_run=args.dry_run)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
