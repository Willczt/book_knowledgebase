import asyncio
from dataclasses import dataclass
from typing import Optional

import structlog

from core.embedding.base import EmbeddingProvider
from vector_store.qdrant_store import QdrantVectorStore, SearchResult

logger = structlog.get_logger()


@dataclass
class RetrievalQuery:
    query_text: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    top_k: int = 10
    score_threshold: float = 0.0
    doc_ids_filter: Optional[list[str]] = None


def _get_sparse_encoder():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")


class HybridRetriever:
    """
    使用 Qdrant Prefetch + RRF Fusion 进行混合检索（dense + sparse BM25）。
    可选接入 CrossEncoderReranker 做二次排序。
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional["CrossEncoderReranker"] = None,
    ) -> None:
        self._store = vector_store
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._sparse_encoder = _get_sparse_encoder()

    async def build_query(
        self,
        query_text: str,
        query_image: Optional[bytes] = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> RetrievalQuery:
        """将原始查询文字（和可选图片）编码为 RetrievalQuery"""
        # Dense embedding
        dense_vector = await self._embedding_provider.embed_query(
            query_text, query_image
        )

        # Sparse BM25 embedding（同步，在线程池执行）
        sparse_results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(self._sparse_encoder.embed([query_text])),
        )
        sparse = sparse_results[0]

        return RetrievalQuery(
            query_text=query_text,
            dense_vector=dense_vector,
            sparse_indices=sparse.indices.tolist(),
            sparse_values=sparse.values.tolist(),
            top_k=top_k,
            score_threshold=score_threshold,
        )

    async def retrieve(
        self,
        collection_name: str,
        query: RetrievalQuery,
    ) -> list[SearchResult]:
        """执行混合检索，返回排序结果"""
        results = await self._store.hybrid_search(
            collection_name=collection_name,
            dense_vector=query.dense_vector,
            sparse_indices=query.sparse_indices,
            sparse_values=query.sparse_values,
            top_k=query.top_k,
            doc_ids_filter=query.doc_ids_filter,
        )

        # 过滤低分结果
        if query.score_threshold > 0:
            results = [r for r in results if r.score >= query.score_threshold]

        # 可选 Reranker
        if self._reranker and len(results) > 1:
            results = await self._reranker.rerank(query.query_text, results)

        return results
