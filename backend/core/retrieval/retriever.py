import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import structlog

from config import get_settings
from core.embedding.base import EmbeddingProvider
from core.retrieval.sparse import embed_sparse
from vector_store.qdrant_store import QdrantVectorStore, SearchResult

if TYPE_CHECKING:  # 仅类型检查时导入，避免与 reranker 形成循环依赖
    from core.retrieval.reranker import Reranker

logger = structlog.get_logger()
settings = get_settings()


@dataclass
class RetrievalQuery:
    query_text: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    top_k: int = 10
    score_threshold: float = 0.0
    doc_ids_filter: Optional[list[str]] = None





class HybridRetriever:
    """
    使用 Qdrant Prefetch + RRF Fusion 进行混合检索（dense + sparse BM25）。
    可选接入 Reranker 做二次排序：先放大候选池到 rerank_candidates，
    再按 rerank 分数重排并把结果截断回 top_k。
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional["Reranker"] = None,
    ) -> None:
        self._store = vector_store
        self._embedding_provider = embedding_provider
        self._reranker = reranker

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

        # Sparse BM25 embedding（同步模型，放线程池避免阻塞事件循环）
        sparse_results = await asyncio.to_thread(embed_sparse, [query_text])
        indices, values = sparse_results[0]

        return RetrievalQuery(
            query_text=query_text,
            dense_vector=dense_vector,
            sparse_indices=indices,
            sparse_values=values,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    async def retrieve(
        self,
        collection_name: str,
        query: RetrievalQuery,
        use_reranker: Optional[bool] = None,
    ) -> list[SearchResult]:
        """执行混合检索，返回排序结果。

        use_reranker：None 表示跟随实例配置（装了 reranker 就用），
        False 显式关闭（例如用户在界面上关掉重排）。
        """
        rerank_enabled = self._reranker is not None and use_reranker is not False

        # 精排需要比 top_k 更大的候选池，否则只是对即将展示的几条换个顺序
        search_top_k = query.top_k
        if rerank_enabled:
            search_top_k = max(query.top_k, settings.rerank_candidates)

        results = await self._store.hybrid_search(
            collection_name=collection_name,
            dense_vector=query.dense_vector,
            sparse_indices=query.sparse_indices,
            sparse_values=query.sparse_values,
            top_k=search_top_k,
            prefetch_limit=max(20, search_top_k * 2),
            doc_ids_filter=query.doc_ids_filter,
        )

        # 过滤低分结果
        if query.score_threshold > 0:
            results = [r for r in results if r.score >= query.score_threshold]

        # 可选 Reranker：只重排序，最终仍截断到用户请求的 top_k
        if rerank_enabled and len(results) > 1:
            results = await self._reranker.rerank(
                query.query_text, results, top_k=query.top_k
            )

        return results
