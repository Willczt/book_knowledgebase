"""rerank 服务（vLLM）的运维接口：状态 / 启动 / 停止 / 重启 / 日志。"""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from core.rerank_server import RerankServerError, get_server

router = APIRouter(prefix="/rerank", tags=["Rerank"])


@router.get("")
@router.get("/status")
async def rerank_status() -> dict[str, Any]:
    """查看 vLLM rerank 服务状态（健康探针 + 进程信息）"""
    return await get_server().status()


@router.post("/start")
async def rerank_start(
    wait: Annotated[bool, Query(description="是否等待模型加载完成（可能数十秒）")] = False,
) -> dict[str, Any]:
    try:
        return await get_server().start(wait=wait)
    except RerankServerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/stop")
async def rerank_stop() -> dict[str, Any]:
    try:
        return await get_server().stop()
    except RerankServerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/restart")
async def rerank_restart(
    wait: Annotated[bool, Query(description="是否等待模型加载完成")] = False,
) -> dict[str, Any]:
    try:
        return await get_server().restart(wait=wait)
    except RerankServerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/logs")
async def rerank_logs(
    lines: Annotated[int, Query(ge=1, le=2000)] = 80,
) -> dict[str, Any]:
    server = get_server()
    status = await server.status()
    return {"log_path": status["log_path"], "lines": server.tail_log(lines)}
