"""RAG 全链路追踪：记录一次查询从输入到输出的每个阶段。

原则：
- 默认关闭（DEBUG_ENABLED=false）：不创建 trace、不额外查库，主链路零负担。
- 分层开关（DEBUG_LAYERS）：query / retrieval / rerank / generation / performance，
  可以只打开关心的一层，未开启的层既不用采集也不返回。
- 只读采集，不改变检索与生成逻辑；调试用的额外查询（向量/BM25 分路分数）失败只记错误。
- 内存保留最近 N 条（DEBUG_MAX_TRACES），可选落盘 storage/traces/<id>.json 供回放。
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import structlog

from config import get_settings
from core.observability.runtime import LAYER_NAMES, get_runtime

logger = structlog.get_logger()

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """粗估 token 数（仅调试展示）：CJK 按 1 字 1 token，其它按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + max(0, (len(text) - cjk + 3) // 4)


def preview(text: Any, limit: int = 200) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


class QueryTrace:
    """一条查询的追踪记录。"""

    def __init__(self, query: str, kind: str = "chat", layers: Optional[str] = None) -> None:
        self.trace_id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.query = query
        self.started_at = datetime.now()
        self._t0 = time.perf_counter()
        raw_layers = layers if layers is not None else ",".join(get_runtime().layers)
        self.layers = {part.strip() for part in (raw_layers or "").split(",") if part.strip()}
        self.stages: list[dict[str, Any]] = []
        self.sections: dict[str, dict[str, Any]] = {name: {} for name in self.layers}
        self.errors: list[dict[str, Any]] = []
        self._timers: dict[str, float] = {}

    # ---------- 分层开关 ----------

    def on(self, layer: str) -> bool:
        return layer in self.layers

    def set(self, layer: str, key: str, value: Any) -> None:
        if layer in self.layers:
            self.sections.setdefault(layer, {})[key] = value

    def update(self, layer: str, data: dict[str, Any]) -> None:
        if layer in self.layers:
            self.sections.setdefault(layer, {}).update(data)

    # ---------- 计时 ----------

    def tic(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def toc(self, name: str, status: str = "ok", error: Optional[str] = None) -> float:
        started = self._timers.pop(name, None)
        ms = (time.perf_counter() - started) * 1000 if started is not None else 0.0
        self.stages.append(
            {"name": name, "duration_ms": round(ms, 1), "status": status, "error": error}
        )
        return ms

    def fail(self, stage: str, exc: BaseException) -> None:
        self.errors.append(
            {
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "at": datetime.now().isoformat(timespec="milliseconds"),
            }
        )

    # ---------- 输出 ----------

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def summary(self) -> dict[str, Any]:
        retrieval = self.sections.get("retrieval") or {}
        rerank = self.sections.get("rerank") or {}
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "query": preview(self.query, 120),
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "total_ms": self.total_ms(),
            "layers": sorted(self.layers),
            "candidates": retrieval.get("candidate_count") or len(retrieval.get("results") or []),
            "reranked": rerank.get("enabled", False),
            "errors": len(self.errors),
        }

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "kind": self.kind,
            # 原始问题单独放 query_text，"query" 键留给 query 层的采集内容
            "query_text": self.query,
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "total_ms": self.total_ms(),
            "layers": sorted(self.layers),
            "errors": self.errors,
        }
        if self.on("performance"):
            data["stages"] = list(self.stages)
        for layer in LAYER_NAMES:
            if self.on(layer) and self.sections.get(layer):
                data[layer] = self.sections[layer]
        return data


def query_text_of(data: dict[str, Any]) -> str:
    """兼容不同版本的 trace：query 可能是原始字符串，也可能已经是 query 层对象。"""
    query = data.get("query")
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        return str(query.get("original") or query.get("rewritten") or "")
    return str(data.get("query_text") or "")


@contextmanager
def span(trace: Optional[QueryTrace], name: str) -> Iterator[None]:
    """给代码块计时，异常时记录错误并继续抛出。trace=None 时零开销。"""
    if trace is None:
        yield
        return
    trace.tic(name)
    try:
        yield
    except Exception as exc:
        trace.toc(name, status="error", error=f"{type(exc).__name__}: {exc}"[:300])
        trace.fail(name, exc)
        raise
    else:
        trace.toc(name)


class TraceStore:
    """最近若干条 trace 的内存队列 + 可选落盘。"""

    def __init__(self) -> None:
        settings = get_settings()
        self._items: deque[dict[str, Any]] = deque(maxlen=max(10, settings.debug_max_traces))
        self._persist = settings.debug_persist
        self._max_files = max(20, settings.debug_max_traces * 2)

    def _dir(self) -> Path:
        root = Path(get_settings().storage_root)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[3] / root
        return root / "traces"

    def _path(self, trace_id: str) -> Optional[Path]:
        return self._dir() / f"{trace_id}.json" if _TRACE_ID_RE.match(trace_id) else None

    async def add(self, trace: QueryTrace) -> dict[str, Any]:
        data = trace.as_dict()
        self._items.append(data)
        if self._persist:
            try:
                path = self._dir() / f"{trace.trace_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                await _write_json(path, data)
                self._prune_files()
            except OSError as exc:  # 落盘失败不影响请求
                logger.warning("trace_persist_failed", trace_id=trace.trace_id, error=str(exc)[:200])
        return data

    def _prune_files(self) -> None:
        try:
            files = sorted(self._dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        for stale in files[self._max_files:]:
            try:
                stale.unlink()
            except OSError:
                continue

    def count(self) -> int:
        return len(self._items)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self._items)[::-1][:limit]
        if items or not self._persist:
            # 列表只返回摘要：完整 trace（含 Prompt / 全量片段）体积太大
            return [self._to_summary(item) for item in items]
        # 内存为空（例如刚重启）时回放磁盘上的历史 trace
        result = []
        try:
            files = sorted(
                self._dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )[:limit]
        except OSError:
            return []
        for path in files:
            data = _read_json(path)
            if data:
                result.append(self._to_summary(data))
        return result

    def get(self, trace_id: str) -> Optional[dict[str, Any]]:
        for item in self._items:
            if item.get("trace_id") == trace_id:
                return item
        path = self._path(trace_id) if self._persist else None
        if path and path.is_file():
            return _read_json(path)
        return None

    async def annotate(self, trace_id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
        """给 trace 追加人工标注（坏 case 归因 / 评估指标），内存与落盘同步更新。"""
        for item in self._items:
            if item.get("trace_id") == trace_id:
                item.update(patch)
                path = self._path(trace_id) if self._persist else None
                if path is not None:
                    await _write_json(path, item)
                return item

        path = self._path(trace_id) if self._persist else None
        if path is not None and path.is_file():
            item = _read_json(path)
            if item is not None:
                item.update(patch)
                await _write_json(path, item)
                return item
        return None

    async def clear(self) -> int:
        removed = len(self._items)
        self._items.clear()
        if self._persist:
            for path in self._dir().glob("*.json"):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def export(self) -> list[dict[str, Any]]:
        items = list(self._items)[::-1]
        if items or not self._persist:
            return items
        return [data for p in sorted(self._dir().glob("*.json")) if (data := _read_json(p))]

    @staticmethod
    def _to_summary(data: dict[str, Any]) -> dict[str, Any]:
        retrieval = data.get("retrieval") or {}
        rerank = data.get("rerank") or {}
        return {
            "trace_id": data.get("trace_id"),
            "kind": data.get("kind"),
            "query": preview(query_text_of(data), 120),
            "started_at": data.get("started_at"),
            "total_ms": data.get("total_ms"),
            "layers": data.get("layers", []),
            "candidates": retrieval.get("candidate_count") or len(retrieval.get("results") or []),
            "reranked": rerank.get("enabled", False),
            "errors": len(data.get("errors") or []),
        }


async def _write_json(path: Path, data: Any) -> None:
    import asyncio

    payload = json.dumps(data, ensure_ascii=False, indent=2)
    await asyncio.to_thread(path.write_text, payload, "utf-8")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_store: Optional[TraceStore] = None


def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore()
    return _store


def reset_trace_store() -> None:
    """测试用：清空单例（不影响磁盘文件）"""
    global _store
    _store = None


def new_trace(query: str, kind: str = "chat") -> Optional[QueryTrace]:
    """调试模式关闭时返回 None，调用方无需分支判断。"""
    if not get_runtime().enabled:
        return None
    return QueryTrace(query, kind=kind)


def percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数（values 需已排序）。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * ratio
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def stage_metrics(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按阶段汇总耗时（P50/P95/最大）与错误数，用于性能面板。"""
    buckets: dict[str, list[float]] = {}
    errors: dict[str, int] = {}
    for item in traces:
        for stage in item.get("stages") or []:
            name = stage.get("name")
            if not name:
                continue
            buckets.setdefault(name, []).append(float(stage.get("duration_ms") or 0.0))
            if stage.get("status") == "error":
                errors[name] = errors.get(name, 0) + 1

    rows = []
    for name, values in buckets.items():
        values.sort()
        rows.append(
            {
                "stage": name,
                "count": len(values),
                "avg_ms": round(sum(values) / len(values), 2),
                "p50_ms": round(percentile(values, 0.50), 2),
                "p95_ms": round(percentile(values, 0.95), 2),
                "max_ms": round(values[-1], 2),
                "errors": errors.get(name, 0),
            }
        )
    return sorted(rows, key=lambda row: row["p95_ms"], reverse=True)


def ranking_metrics(
    ranked_ids: list[str], relevant_ids: list[str], k: int = 10
) -> dict[str, Any]:
    """基于人工标注的标准片段计算 Recall@K / MRR / NDCG@K / Precision@K。"""
    relevant = set(relevant_ids or [])
    top = ranked_ids[: max(1, k)]
    if not relevant:
        return {"k": k, "relevant": 0, "note": "未提供标准片段，无法计算"}

    hits = [1 if chunk_id in relevant else 0 for chunk_id in top]
    recall = sum(hits) / len(relevant)
    precision = (sum(hits) / len(top)) if top else 0.0
    reciprocal_rank = next(
        (1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0
    )
    dcg = sum(1.0 / math.log2(index + 2) for index, hit in enumerate(hits) if hit)
    ideal_hits = min(len(relevant), len(top))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        "k": k,
        "relevant": len(relevant),
        "hits": sum(hits),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mrr": round(reciprocal_rank, 4),
        "ndcg": round(dcg / idcg, 4) if idcg else 0.0,
        "ranked_ids": top,
    }


async def record_trace(trace: Optional[QueryTrace]) -> Optional[str]:
    if trace is None:
        return None
    try:
        await get_trace_store().add(trace)
    except Exception as exc:  # 记录失败不影响主链路
        logger.warning("trace_record_failed", error=str(exc)[:200])
        return trace.trace_id
    return trace.trace_id
