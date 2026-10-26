"""检索结果的二次排序（rerank）。

两种后端，通过 RERANK_BACKEND 切换：
- http（默认）：调用 vLLM 的 /v1/rerank 接口，默认模型 BAAI/bge-reranker-v2-m3
  （中文效果明显好于本地英文小模型，方案与 ebook_knowledgebase 项目一致）；
- flashrank：本地 ONNX cross-encoder，无外部服务依赖，但模型偏英文场景。

任一后端失败都只记日志并回退到原始顺序，绝不影响问答主链路；
连续失败还会短暂熔断，避免每条消息都为不可用的服务付出连接/超时开销。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Protocol

import httpx
import structlog

from config import get_settings
from vector_store.qdrant_store import SearchResult

logger = structlog.get_logger()

# rerank 失败后的冷却时间（秒）：期间直接走原顺序，不再发请求
_FAILURE_COOLDOWN_SECONDS = 60.0


class Reranker(Protocol):
    """重排器接口：按相关度重排检索结果，返回条数不超过 top_k。"""

    @property
    def model_name(self) -> str: ...

    async def rerank(
        self,
        query_text: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]: ...


def _parse_rerank_scores(body: Any, expected: int) -> list[float]:
    """解析 /v1/rerank 响应，按 index 对齐回填分数（缺失的补 0）。"""
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        raise ValueError("rerank 响应缺少 results 列表")

    scores = [0.0] * expected
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < expected:
            continue
        try:
            scores[index] = float(item.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            continue
    return scores


class HttpReranker:
    """通过 HTTP 调用 vLLM 的 cross-encoder rerank 接口。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        self._cooldown_until = 0.0

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self,
        query_text: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        if len(results) < 2:
            return results[:top_k] if top_k else results
        if time.monotonic() < self._cooldown_until:
            return results[:top_k] if top_k else results

        started = time.perf_counter()
        try:
            scores = await self._score(query_text, [r.text for r in results])
        except Exception as exc:  # 网络/协议/服务端错误一律降级
            self._cooldown_until = time.monotonic() + _FAILURE_COOLDOWN_SECONDS
            logger.warning(
                "rerank_failed",
                backend="http",
                model=self._model,
                error=str(exc)[:200],
            )
            return results[:top_k] if top_k else results

        order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        # score 保留融合分，rerank 分单独带出（展示时优先用它，顺序与百分比才一致）
        reranked = []
        for index in order:
            item = results[index]
            item.rerank_score = scores[index]
            reranked.append(item)
        logger.info(
            "rerank_applied",
            backend="http",
            model=self._model,
            candidates=len(results),
            top_score=round(scores[order[0]], 4),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return reranked[:top_k] if top_k else reranked

    async def _score(self, query: str, documents: list[str]) -> list[float]:
        payload = {"model": self._model, "query": query, "documents": documents}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url}/rerank"

        if self._client is not None:
            response = await self._client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)

        response.raise_for_status()
        return _parse_rerank_scores(response.json(), len(documents))


class FlashRankReranker:
    """本地 FlashRank（ONNX cross-encoder），无 GPU、无外部服务依赖。"""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        cache_dir: str = "/tmp/flashrank",
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._ranker = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._ranker is None:
            from flashrank import Ranker

            self._ranker = Ranker(model_name=self._model_name, cache_dir=self._cache_dir)

    def _sync_rerank(
        self, query_text: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        self._ensure_loaded()
        from flashrank import RerankRequest

        passages = [{"id": r.chunk_id, "text": r.text} for r in results]
        reranked = self._ranker.rerank(RerankRequest(query=query_text, passages=passages))

        id_to_result = {r.chunk_id: r for r in results}
        ordered = []
        for item in reranked:
            chunk_id = item.get("id") or item.get("text", "")[:50]
            original = id_to_result.get(chunk_id)
            if original is not None:
                score = item.get("score")
                original.rerank_score = float(score) if score is not None else None
                ordered.append(original)
        return ordered

    async def rerank(
        self,
        query_text: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        if len(results) < 2:
            return results[:top_k] if top_k else results

        started = time.perf_counter()
        try:
            reranked = await asyncio.to_thread(self._sync_rerank, query_text, results)
        except Exception as exc:
            logger.warning(
                "rerank_failed",
                backend="flashrank",
                model=self._model_name,
                error=str(exc)[:200],
            )
            return results[:top_k] if top_k else results

        logger.info(
            "rerank_applied",
            backend="flashrank",
            model=self._model_name,
            candidates=len(results),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return reranked[:top_k] if top_k else reranked


_reranker: Optional[Reranker] = None


def build_reranker() -> Reranker:
    """按当前配置创建 reranker（不缓存，便于测试/热切换）。"""
    s = get_settings()
    backend = (s.rerank_backend or "http").strip().lower()
    if backend == "flashrank":
        return FlashRankReranker(s.rerank_local_model)
    return HttpReranker(
        base_url=s.rerank_base_url,
        model=s.rerank_model,
        api_key=s.rerank_api_key,
        timeout=s.rerank_timeout,
    )


def get_reranker() -> Reranker:
    """进程内共享的 reranker（HTTP 客户端/本地模型只建一次）"""
    global _reranker
    if _reranker is None:
        _reranker = build_reranker()
    return _reranker


def reset_reranker() -> None:
    """清空进程内单例（测试用）"""
    global _reranker
    _reranker = None
