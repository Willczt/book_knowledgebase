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
from typing import TYPE_CHECKING, Any, Optional, Protocol

import httpx
import structlog

from config import get_settings
from vector_store.qdrant_store import SearchResult

if TYPE_CHECKING:
    from core.observability.trace import QueryTrace

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
        trace: Optional["QueryTrace"] = None,
    ) -> list[SearchResult]: ...


def _record_rerank(
    trace: Optional["QueryTrace"],
    *,
    backend: str,
    model: str,
    results: list[SearchResult],
    status: str = "applied",
    order: Optional[list[int]] = None,
    scores: Optional[list[float]] = None,
    duration_ms: int = 0,
    error: Optional[str] = None,
) -> None:
    """把一次重排的输入/输出顺序与分数写入 trace。"""
    if trace is None or not trace.on("rerank"):
        return
    data: dict[str, Any] = {
        "enabled": status == "applied",
        "status": status,
        "backend": backend,
        "model": model,
        "candidates": len(results),
        "duration_ms": duration_ms,
        "order_before": [r.chunk_id for r in results],
    }
    if order is not None:
        data["order_after"] = [results[i].chunk_id for i in order]
    if scores is not None and order is not None:
        data["scores"] = {results[i].chunk_id: round(scores[i], 6) for i in order}
        data["top_score"] = round(scores[order[0]], 6) if order else None
        # 每个候选的「进重排前分数/排名」→「重排后分数/排名」，用于判断重排是否把正确答案挤后
        data["candidates_detail"] = [
            {
                "chunk_id": results[i].chunk_id,
                "rank_before": i + 1,
                "rank_after": rank,
                "score_before": round(results[i].score, 6),
                "score_after": round(scores[i], 6),
            }
            for rank, i in enumerate(order, 1)
        ]
    if error:
        data["error"] = error
    trace.update("rerank", data)


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
        trace: Optional["QueryTrace"] = None,
    ) -> list[SearchResult]:
        if len(results) < 2:
            _record_rerank(
                trace,
                backend="http",
                model=self._model,
                results=results,
                status="skipped",
            )
            return results[:top_k] if top_k else results
        if time.monotonic() < self._cooldown_until:
            _record_rerank(
                trace,
                backend="http",
                model=self._model,
                results=results,
                status="cooldown",
                error="rerank 服务处于失败冷却期",
            )
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
            _record_rerank(
                trace,
                backend="http",
                model=self._model,
                results=results,
                status="failed",
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            return results[:top_k] if top_k else results

        order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        # score 保留融合分，rerank 分单独带出（展示时优先用它，顺序与百分比才一致）
        reranked = []
        for index in order:
            item = results[index]
            item.rerank_score = scores[index]
            reranked.append(item)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _record_rerank(
            trace,
            backend="http",
            model=self._model,
            results=results,
            order=order[:top_k] if top_k else order,
            scores=scores,
            duration_ms=elapsed_ms,
        )
        logger.info(
            "rerank_applied",
            backend="http",
            model=self._model,
            candidates=len(results),
            top_score=round(scores[order[0]], 4),
            elapsed_ms=elapsed_ms,
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
    ) -> tuple[list[SearchResult], list[int], list[float]]:
        """返回 (重排后的结果, 原始下标顺序, 与原始下标对齐的分数)。"""
        self._ensure_loaded()
        from flashrank import RerankRequest

        passages = [{"id": r.chunk_id, "text": r.text} for r in results]
        reranked = self._ranker.rerank(RerankRequest(query=query_text, passages=passages))

        id_to_index = {r.chunk_id: i for i, r in enumerate(results)}
        scores = [0.0] * len(results)
        order: list[int] = []
        ordered: list[SearchResult] = []
        for item in reranked:
            chunk_id = item.get("id") or item.get("text", "")[:50]
            index = id_to_index.get(chunk_id)
            if index is None:
                continue
            score = item.get("score")
            value = float(score) if score is not None else None
            results[index].rerank_score = value
            scores[index] = value if value is not None else 0.0
            order.append(index)
            ordered.append(results[index])
        return ordered, order, scores

    async def rerank(
        self,
        query_text: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
        trace: Optional["QueryTrace"] = None,
    ) -> list[SearchResult]:
        if len(results) < 2:
            _record_rerank(
                trace,
                backend="flashrank",
                model=self._model_name,
                results=results,
                status="skipped",
            )
            return results[:top_k] if top_k else results

        started = time.perf_counter()
        try:
            reranked, order, scores = await asyncio.to_thread(
                self._sync_rerank, query_text, results
            )
        except Exception as exc:
            logger.warning(
                "rerank_failed",
                backend="flashrank",
                model=self._model_name,
                error=str(exc)[:200],
            )
            _record_rerank(
                trace,
                backend="flashrank",
                model=self._model_name,
                results=results,
                status="failed",
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            return results[:top_k] if top_k else results

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _record_rerank(
            trace,
            backend="flashrank",
            model=self._model_name,
            results=results,
            order=order[:top_k] if top_k else order,
            scores=scores,
            duration_ms=elapsed_ms,
        )
        logger.info(
            "rerank_applied",
            backend="flashrank",
            model=self._model_name,
            candidates=len(results),
            elapsed_ms=elapsed_ms,
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
