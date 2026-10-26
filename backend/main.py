import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from api.v1.router import api_router
from config import get_settings
from core.exceptions import BookRAGException, ProviderUnavailable
from core.logging import configure_logging
from core.middleware import RequestContextMiddleware
from db.engine import AsyncSessionLocal, dispose_db, init_db
from tasks.worker import get_worker

settings = get_settings()
logger = structlog.get_logger()

# 探针超时：探针必须快速返回，慢依赖不能让编排器误判为进程死亡
_PROBE_TIMEOUT = 3.0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    logger.info(
        "application_starting",
        env=settings.log_format,
        database="sqlite" if settings.is_sqlite else "postgres",
        ingest_workers=settings.ingest_workers,
    )

    await init_db()

    # 存储目录
    for subdir in ["docs", "images"]:
        Path(settings.storage_root, subdir).mkdir(parents=True, exist_ok=True)

    # 摄入 worker：启动时先恢复租约过期的任务，再开始消费队列
    worker = get_worker()
    await worker.start()

    # 可选：顺带拉起 vLLM rerank 服务（后台任务，不阻塞 API 启动）
    autostart: asyncio.Task | None = None
    if settings.rerank_enabled and settings.rerank_autostart:
        autostart = asyncio.create_task(_autostart_rerank())

    logger.info("application_started", storage_root=settings.storage_root)
    try:
        yield
    finally:
        logger.info("application_stopping")
        if autostart is not None:
            autostart.cancel()
            try:
                await autostart
            except (asyncio.CancelledError, Exception):
                pass
            await _stop_rerank()

        # 先停 worker（等在跑任务收尾），再关连接池
        await worker.stop()

        from api.deps import close_qdrant

        await close_qdrant()
        await dispose_db()
        logger.info("application_shutdown")


async def _autostart_rerank() -> None:
    from core.rerank_server import RerankServerError, get_server

    try:
        state = await get_server().start(wait=False)
        logger.info(
            "rerank_autostart",
            message=state.get("message"),
            pid=state.get("pid"),
        )
    except RerankServerError as exc:
        logger.warning("rerank_autostart_failed", error=str(exc))
    except Exception as exc:  # 自启失败不影响主服务
        logger.warning("rerank_autostart_failed", error=str(exc)[:200])


async def _stop_rerank() -> None:
    from core.rerank_server import get_server

    try:
        state = await get_server().stop()
        logger.info("rerank_autostop", message=state.get("message"))
    except Exception as exc:
        logger.warning("rerank_autostop_failed", error=str(exc)[:200])


app = FastAPI(
    title="电子书 RAG 知识库 API",
    description="支持 epub/pdf/markdown/txt 的多模态 RAG 系统，混合检索，多模型支持",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------- 探针 ----------

@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    """存活探针：只证明进程还能响应，不查依赖。"""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz():
    """就绪探针：DB + Qdrant 均可用才返回 200，否则 503（负载均衡应摘流量）。"""
    checks: dict[str, str] = {}
    healthy = True

    try:
        async with AsyncSessionLocal() as db:
            await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=_PROBE_TIMEOUT)
        checks["database"] = "ok"
    except Exception as exc:
        healthy = False
        checks["database"] = f"{type(exc).__name__}: {exc}"[:120]

    try:
        from api.deps import get_qdrant

        await asyncio.wait_for(
            get_qdrant().client.get_collections(), timeout=_PROBE_TIMEOUT
        )
        checks["qdrant"] = "ok"
    except Exception as exc:
        healthy = False
        checks["qdrant"] = f"{type(exc).__name__}: {exc}"[:120]

    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "checks": checks},
        status_code=200 if healthy else 503,
    )


# ---------- 异常处理 ----------

@app.exception_handler(ProviderUnavailable)
async def _provider_unavailable_handler(request: Request, exc: ProviderUnavailable):
    """模型服务不可用（未启动 / 超时 / 鉴权失败）统一返回可读提示，而不是 500"""
    logger.warning("provider_unavailable", path=str(request.url.path), error=str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(BookRAGException)
async def _bookrag_exception_handler(request: Request, exc: BookRAGException):
    logger.warning("bookrag_exception", path=str(request.url.path), error=str(exc))
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """兜底：不把堆栈回给客户端，但要把 request_id 带回去方便查日志。"""
    context = structlog.contextvars.get_contextvars()
    request_id = context.get("request_id")
    logger.exception("unhandled_exception", path=str(request.url.path))
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "request_id": request_id},
    )


# ---------- 中间件 ----------
# 注意顺序：后添加的中间件先执行。RequestContext 放最后 → 最外层，
# 这样 CORS 头、鉴权失败响应、访问日志都能被统一处理。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestContextMiddleware, api_token=settings.api_token)

app.include_router(api_router)

# 提供图片静态文件访问
images_dir = Path(settings.storage_root) / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
