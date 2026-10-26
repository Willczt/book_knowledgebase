import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import structlog

from config import get_settings
from core.embedding.base import EmbeddingProvider
from core.observability.trace import span
from core.retrieval.sparse import embed_sparse
from vector_store.qdrant_store import QdrantVectorStore, SearchResult

if TYPE_CHECKING:  # 仅类型检查时导入，避免与 reranker 形成循环依赖
    from core.observability.trace import QueryTrace
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
    # 精排候选池大小（None 跟随全局配置），调试时可临时调整对比效果
    rerank_candidates: Optional[int] = None





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
        rerank_candidates: Optional[int] = None,
        trace: Optional["QueryTrace"] = None,
    ) -> RetrievalQuery:
        """将原始查询文字（和可选图片）编码为 RetrievalQuery"""
        # Dense embedding
        with span(trace, "embedding"):
            dense_vector = await self._embedding_provider.embed_query(
                query_text, query_image
            )

        # Sparse BM25 embedding（同步模型，放线程池避免阻塞事件循环）
        with span(trace, "sparse_embedding"):
            sparse_results = await asyncio.to_thread(embed_sparse, [query_text])
        indices, values = sparse_results[0]

        if trace is not None:
            model_info = getattr(self._embedding_provider, "model_info", None)
            provider_name = getattr(model_info, "provider", None) or getattr(
                self._embedding_provider, "provider", type(self._embedding_provider).__name__
            )
            model_id = getattr(model_info, "model_id", None) or getattr(
                self._embedding_provider, "model_id", None
            )
            trace.update(
                "query",
                {
                    "retrieval_strategy": "hybrid(dense + sparse, RRF)",
                    "params": {
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "rerank_candidates": rerank_candidates or settings.rerank_candidates,
                        "embedding_provider": provider_name,
                        "embedding_model": model_id,
                    },
                },
            )
            trace.set(
                "query",
                "embedding",
                {
                    "provider": provider_name,
                    "model": model_id,
                    "dim": len(dense_vector),
                    # 只存前若干维，足够判断向量是否异常（全 0 / NaN / 分布异常）
                    "head": [round(float(x), 6) for x in dense_vector[:16]],
                    "norm": round(sum(float(x) ** 2 for x in dense_vector) ** 0.5, 6),
                    "sparse_terms": len(indices),
                    "sparse_top_terms": [
                        {"index": int(i), "weight": round(float(v), 6)}
                        for i, v in list(zip(indices, values, strict=True))[:10]
                    ],
                    "has_image": query_image is not None,
                },
            )

        return RetrievalQuery(
            query_text=query_text,
            dense_vector=dense_vector,
            sparse_indices=indices,
            sparse_values=values,
            top_k=top_k,
            score_threshold=score_threshold,
            rerank_candidates=rerank_candidates,
        )

    async def retrieve(
        self,
        collection_name: str,
        query: RetrievalQuery,
        use_reranker: Optional[bool] = None,
        trace: Optional["QueryTrace"] = None,
    ) -> list[SearchResult]:
        """执行混合检索，返回排序结果。

        use_reranker：None 表示跟随实例配置（装了 reranker 就用），
        False 显式关闭（例如用户在界面上关掉重排）。
        trace：调试模式下采集召回/过滤/重排的片段级明细，None 时零开销。
        """
        rerank_enabled = self._reranker is not None and use_reranker is not False
        rerank_candidates = query.rerank_candidates or settings.rerank_candidates

        # 精排需要比 top_k 更大的候选池，否则只是对即将展示的几条换个顺序
        search_top_k = query.top_k
        if rerank_enabled:
            search_top_k = max(query.top_k, rerank_candidates)
        prefetch_limit = max(20, search_top_k * 2)

        trace_retrieval = trace is not None and trace.on("retrieval")
        debug_sink: Optional[dict] = {} if trace_retrieval else None

        with span(trace, "qdrant_search"):
            results = await self._store.hybrid_search(
                collection_name=collection_name,
                dense_vector=query.dense_vector,
                sparse_indices=query.sparse_indices,
                sparse_values=query.sparse_values,
                top_k=search_top_k,
                prefetch_limit=prefetch_limit,
                doc_ids_filter=query.doc_ids_filter,
                debug_sink=debug_sink,
            )

        candidate_count = len(results)

        # 过滤低分结果（被过滤的片段同样记进 trace，方便判断阈值是否过严）
        filtered_out: list[SearchResult] = []
        if query.score_threshold > 0:
            filtered_out = [r for r in results if r.score < query.score_threshold]
            results = [r for r in results if r.score >= query.score_threshold]

        # 可选 Reranker：只重排序，最终仍截断到用户请求的 top_k
        if rerank_enabled and len(results) > 1:
            with span(trace, "rerank"):
                results = await self._reranker.rerank(
                    query.query_text, results, trace=trace
                )
        elif trace is not None and trace.on("rerank"):
            # 没走到 rerank() 时也要留一条记录，否则 trace 的 rerank 段整段缺失，
            # 前端只能显示「未开启重排层或未触发重排」，无法定位是配置关了还是服务不可用
            from core.retrieval.reranker import record_rerank_skip

            backend_name = (settings.rerank_backend or "http").strip().lower()
            model_name = getattr(self._reranker, "model_name", None) or settings.rerank_model
            if use_reranker is False:
                status = "disabled"
                reason = (
                    "本次请求未启用重排（use_reranker=false）："
                    "可能来自界面开关，或 RERANK_ENABLED=false；结果按融合分排序"
                )
            elif self._reranker is None:
                status = "unavailable"
                reason = "未构建 reranker：检查 RERANK_ENABLED 与 RERANK_BACKEND 配置"
            else:
                status = "skipped"
                reason = (
                    f"过滤后仅剩 {len(results)} 条候选"
                    f"（score_threshold={query.score_threshold:g}），少于 2 条无需重排"
                )
            record_rerank_skip(
                trace,
                backend=backend_name,
                model=model_name,
                results=results,
                status=status,
                reason=reason,
            )

        final = results[: query.top_k]
        dropped = results[query.top_k:]

        if trace_retrieval:
            data = debug_sink or {}
            trace.update(
                "retrieval",
                {
                    "collection": collection_name,
                    "params": {
                        "top_k": query.top_k,
                        "search_top_k": search_top_k,
                        "prefetch_limit": prefetch_limit,
                        "score_threshold": query.score_threshold,
                        "use_reranker": rerank_enabled,
                        "rerank_candidates": rerank_candidates,
                        "doc_ids_filter": query.doc_ids_filter,
                    },
                    "candidate_count": candidate_count,
                    "filtered_count": len(filtered_out),
                    "threshold": query.score_threshold,
                    "returned": len(final),
                    "fusion": data.get("fusion"),
                    # 本系统未做检索结果缓存，明确记录以免误判
                    "cache": {"enabled": False, "hit": False, "note": "未启用检索缓存"},
                    "vector_hits": len(data.get("vector") or {}),
                    "sparse_hits": len(data.get("sparse") or {}),
                    "warnings": self._retrieval_warnings(data, query),
                    "errors": {
                        key: value
                        for key, value in data.items()
                        if key.endswith("_error")
                    },
                },
            )
            trace.set(
                "retrieval",
                "results",
                [
                    *(
                        self._trace_row(r, data, True, None, rank)
                        for rank, r in enumerate(final, 1)
                    ),
                    *(
                        self._trace_row(r, data, False, "beyond_top_k", None)
                        for r in dropped
                    ),
                    *(
                        self._trace_row(r, data, False, "below_score_threshold", None)
                        for r in filtered_out
                    ),
                ],
            )

        return final

    @staticmethod
    def _retrieval_warnings(debug: dict, query: RetrievalQuery) -> list[str]:
        """把「某一路完全没命中」这类问题显式标出来，避免只看到融合分却不知原因。"""
        warnings = []
        if query.sparse_indices and not debug.get("sparse") and "sparse_error" not in debug:
            warnings.append(
                "稀疏(BM25)检索 0 命中：查询被切成 "
                f"{len(query.sparse_indices)} 个 term，可能是分词与索引不一致，"
                "混合检索实际退化成了纯向量检索"
            )
        if not debug.get("vector") and "vector_error" not in debug:
            warnings.append("稠密检索 0 命中：确认 collection 是否有数据 / 向量维度是否一致")
        return warnings

    @staticmethod
    def _trace_row(
        result: SearchResult,
        debug: dict,
        included: bool,
        drop_reason: Optional[str],
        final_rank: Optional[int],
    ) -> dict:
        """一个片段的调试明细：各路分数、原始排名、是否进入最终上下文。"""
        vector = (debug.get("vector") or {}).get(result.chunk_id) or {}
        sparse = (debug.get("sparse") or {}).get(result.chunk_id) or {}
        return {
            "chunk_id": result.chunk_id,
            "text": result.text,
            "text_preview": result.text[:200],
            "score": round(result.score, 6),
            "rerank_score": None
            if result.rerank_score is None
            else round(result.rerank_score, 6),
            "vector_score": vector.get("score"),
            "vector_rank": vector.get("rank"),
            "sparse_score": sparse.get("score"),
            "sparse_rank": sparse.get("rank"),
            "source_doc": result.source_doc,
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "chunk_type": result.chunk_type,
            "image_path": result.image_path,
            "metadata": result.metadata or {},
            "included": included,
            "drop_reason": drop_reason,
            "final_rank": final_rank,
        }
