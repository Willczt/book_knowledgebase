import asyncio
from typing import Optional

import structlog

from vector_store.qdrant_store import SearchResult

logger = structlog.get_logger()


class CrossEncoderReranker:
    """
    可选的 cross-encoder 二次排序器。
    使用 FlashRank（ms-marco-MiniLM-L-6-v2），纯本地 ONNX 运行，无 GPU 需求。
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._ranker = None

    def _ensure_loaded(self):
        if self._ranker is None:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self._model_name, cache_dir="/tmp/flashrank")

    async def rerank(
        self,
        query_text: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        if not results:
            return results

        try:
            def _sync_rerank():
                self._ensure_loaded()
                from flashrank import RerankRequest
                passages = [{"id": r.chunk_id, "text": r.text} for r in results]
                request = RerankRequest(query=query_text, passages=passages)
                reranked = self._ranker.rerank(request)
                # 按 rerank score 重排，映射回原 SearchResult
                id_to_result = {r.chunk_id: r for r in results}
                reranked_results = []
                for item in reranked:
                    chunk_id = item.get("id") or item.get("text", "")[:50]
                    original = id_to_result.get(chunk_id)
                    if original:
                        original.score = item.get("score", original.score)
                        reranked_results.append(original)
                return reranked_results

            reranked = await asyncio.to_thread(_sync_rerank)
            return reranked[:top_k] if top_k else reranked

        except Exception as e:
            logger.warning("reranker_failed", error=str(e))
            return results


_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """进程内共享的 reranker（模型只加载一次）"""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
