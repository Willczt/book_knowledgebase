"""RAG 全链路调试接口。

- 运行时开关：是否采集、采集哪几层（热改，无需重启）
- trace 查询：列表 / 详情 / 导出 / 清空
- 性能汇总：各阶段 P50 / P95 / 错误数
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.observability.runtime import LAYER_LABELS, LAYER_NAMES, get_runtime
from core.observability.trace import get_trace_store, ranking_metrics, stage_metrics

router = APIRouter(prefix="/debug", tags=["Debug"])


class DebugConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    layers: Optional[list[str]] = Field(default=None, description="query/retrieval/rerank/generation/performance")


class TraceLabel(BaseModel):
    """坏 case 归因：哪个环节导致失败。"""

    verdict: str = Field(..., pattern="^(good|bad)$")
    stage: Optional[str] = Field(default=None, description="query/retrieval/rerank/generation")
    note: Optional[str] = None


class TraceEvaluation(BaseModel):
    """离线评估：给出标准相关片段，计算排序指标。"""

    relevant_chunk_ids: list[str] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=50)
    note: Optional[str] = None


@router.get("/status")
async def debug_status() -> dict:
    """当前调试开关、可用层、已缓存 trace 数量。"""
    data = get_runtime().snapshot()
    data["trace_count"] = get_trace_store().count()
    data["layer_labels"] = LAYER_LABELS
    return data


@router.put("/status")
async def update_debug_status(body: DebugConfigUpdate) -> dict:
    """热改调试开关与分层开关，从下一个请求开始生效。"""
    if body.layers is not None:
        invalid = [name for name in body.layers if name not in LAYER_NAMES]
        if invalid:
            raise HTTPException(status_code=400, detail=f"未知的调试层：{invalid}")
    snapshot = get_runtime().configure(enabled=body.enabled, layers=body.layers)
    snapshot["trace_count"] = get_trace_store().count()
    return snapshot


@router.get("/traces")
async def list_traces(limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
    """最近的 trace 摘要（按时间倒序）。"""
    return get_trace_store().list(limit=limit)


@router.delete("/traces")
async def clear_traces() -> dict:
    """清空内存与磁盘上的 trace。"""
    removed = await get_trace_store().clear()
    return {"removed": removed}


@router.get("/metrics")
async def debug_metrics(limit: int = Query(default=100, ge=1, le=200)) -> dict:
    """各阶段耗时分位数与错误数，用于发现性能滑坡。"""
    traces = get_trace_store().export()[:limit]
    return {
        "traces": len(traces),
        "stages": stage_metrics(traces),
    }


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict:
    data = get_trace_store().get(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="trace 不存在或已被清理")
    return data


@router.post("/traces/{trace_id}/label")
async def label_trace(trace_id: str, body: TraceLabel) -> dict:
    """给一次请求打标（好/坏 + 出问题的环节），用于后续回归对比。"""
    annotation = {
        "verdict": body.verdict,
        "stage": body.stage,
        "note": body.note,
    }
    data = await get_trace_store().annotate(trace_id, {"label": annotation})
    if data is None:
        raise HTTPException(status_code=404, detail="trace 不存在或已被清理")
    return {"trace_id": trace_id, "label": annotation}


@router.post("/traces/{trace_id}/evaluate")
async def evaluate_trace(trace_id: str, body: TraceEvaluation) -> dict:
    """用人工给出的相关片段计算 Recall@K / MRR / NDCG@K，并写回 trace。"""
    data = get_trace_store().get(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="trace 不存在或已被清理")

    ranked_ids = _ranked_chunk_ids(data)
    metrics = ranking_metrics(ranked_ids, body.relevant_chunk_ids, k=body.k)
    metrics["note"] = body.note
    # 生成阶段的引用覆盖率，作为「回答是否落到检索片段上」的弱指标
    citations = (data.get("generation") or {}).get("citations") or []
    if citations:
        cited = [item for item in citations if item.get("cited")]
        metrics["citation_coverage"] = round(len(cited) / len(citations), 4)

    await get_trace_store().annotate(trace_id, {"quality": metrics})
    return {"trace_id": trace_id, "quality": metrics}


def _ranked_chunk_ids(trace: dict) -> list[str]:
    """按最终进入上下文的名次取出片段 ID（没有名次时退回重排顺序）。"""
    results = (trace.get("retrieval") or {}).get("results") or []
    included = [row for row in results if row.get("included")]
    if included:
        included.sort(key=lambda row: row.get("final_rank") or 10**6)
        return [row.get("chunk_id") for row in included if row.get("chunk_id")]
    return list(((trace.get("rerank") or {}).get("order_after") or []))


@router.get("/traces/{trace_id}/export")
async def export_trace(trace_id: str) -> JSONResponse:
    """导出单条 trace 为 JSON 文件，便于离线分析坏 case。"""
    data = get_trace_store().get(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="trace 不存在或已被清理")
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f'attachment; filename="trace-{trace_id}.json"'},
    )
