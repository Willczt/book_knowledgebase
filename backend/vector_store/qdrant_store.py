import uuid
from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Prefetch,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import get_settings

settings = get_settings()


@dataclass
class ChunkPoint:
    text: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    doc_id: str
    source_doc: str
    chunk_type: str = "text"         # "text" | "image_caption"
    image_path: Optional[str] = None
    page_num: Optional[int] = None
    chunk_index: int = 0
    extra_metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    source_doc: str
    doc_id: str
    chunk_type: str
    page_num: Optional[int] = None
    image_path: Optional[str] = None


class QdrantVectorStore:
    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            kwargs: dict = {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "check_compatibility": False,
            }
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            self._client = AsyncQdrantClient(**kwargs)
        return self._client

    async def create_collection(self, collection_name: str, vector_dim: int) -> None:
        """创建含 dense + sparse 双向量的 Qdrant 集合"""
        exists = await self.client.collection_exists(collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
            )

    async def upsert_points(self, collection_name: str, points: list[ChunkPoint]) -> None:
        """批量写入 chunk 向量点"""
        qdrant_points = []
        for p in points:
            qdrant_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": p.dense_vector,
                        "sparse": SparseVector(
                            indices=p.sparse_indices,
                            values=p.sparse_values,
                        ),
                    },
                    payload={
                        "text": p.text,
                        "doc_id": p.doc_id,
                        "source_doc": p.source_doc,
                        "chunk_type": p.chunk_type,
                        "image_path": p.image_path,
                        "page_num": p.page_num,
                        "chunk_index": p.chunk_index,
                        **p.extra_metadata,
                    },
                )
            )
        if qdrant_points:
            await self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=True,
            )

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 10,
        prefetch_limit: int = 20,
        doc_ids_filter: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """RRF 混合检索：dense + sparse prefetch → RRF Fusion"""
        query_filter = None
        if doc_ids_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(any=doc_ids_filter))
                ]
            )

        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        search_results = []
        for point in results.points:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=str(point.id),
                    text=payload.get("text", ""),
                    score=point.score,
                    source_doc=payload.get("source_doc", ""),
                    doc_id=payload.get("doc_id", ""),
                    chunk_type=payload.get("chunk_type", "text"),
                    page_num=payload.get("page_num"),
                    image_path=payload.get("image_path"),
                )
            )
        return search_results

    async def delete_doc_points(self, collection_name: str, doc_id: str) -> None:
        """删除指定文档的所有向量点"""
        await self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    async def clear_collection(self, collection_name: str, vector_dim: int) -> None:
        """清空集合所有数据（保留集合结构）"""
        # 先删除再重建，最简洁
        await self.client.delete_collection(collection_name)
        await self.create_collection(collection_name, vector_dim)

    async def delete_collection(self, collection_name: str) -> None:
        exists = await self.client.collection_exists(collection_name)
        if exists:
            await self.client.delete_collection(collection_name)

    async def count_points(self, collection_name: str) -> int:
        try:
            info = await self.client.get_collection(collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
