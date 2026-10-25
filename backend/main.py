from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.v1.router import api_router
from config import get_settings
from core.exceptions import BookRAGException, ProviderUnavailable
from db.engine import init_db

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动时初始化数据库
    await init_db()

    # 清理上次进程中断遗留的摄入任务
    await _cleanup_stale_documents()

    # 确保存储目录存在
    for subdir in ["docs", "images"]:
        Path(settings.storage_root, subdir).mkdir(parents=True, exist_ok=True)

    logger.info("application_started", storage_root=settings.storage_root)
    yield
    logger.info("application_shutdown")


async def _cleanup_stale_documents() -> None:
    """
    摄入任务跑在进程内，服务重启后任务丢失，文档会永久停留在 pending / processing。

    通过心跳文件里的 PID 判断任务是否还活着，只把「确实没人跑」的标记为 failed，
    这样多 worker 部署时不会误杀其他 worker 正在处理的任务。
    """
    if not settings.cleanup_stale_documents:
        return

    from db import crud
    from db.engine import AsyncSessionLocal
    from tasks import liveness

    async with AsyncSessionLocal() as db:
        unfinished = await crud.list_unfinished_documents(db)
        if not unfinished:
            return

        candidates = [doc.id for doc in unfinished]
        alive = liveness.live_doc_ids(candidates)
        stale_ids = [doc_id for doc_id in candidates if doc_id not in alive]

        if not stale_ids:
            return

        count = await crud.fail_documents(
            db,
            stale_ids,
            "摄入任务已中断（进程退出或超时），请删除后重新上传该文档",
        )

    liveness.clear_all(stale_ids)
    logger.warning(
        "stale_documents_marked_failed",
        count=count,
        doc_ids=stale_ids[:10],
        still_running=len(alive),
    )


app = FastAPI(
    title="电子书 RAG 知识库 API",
    description="支持 epub/pdf/markdown/txt 的多模态 RAG 系统，混合检索，多模型支持",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ProviderUnavailable)
async def _provider_unavailable_handler(request: Request, exc: ProviderUnavailable):
    """模型服务不可用（未启动 / 超时 / 鉴权失败）统一返回可读提示，而不是 500"""
    logger.warning("provider_unavailable", path=str(request.url.path), error=str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(BookRAGException)
async def _bookrag_exception_handler(request: Request, exc: BookRAGException):
    logger.warning("bookrag_exception", path=str(request.url.path), error=str(exc))
    return JSONResponse(status_code=400, content={"detail": str(exc)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# 提供图片静态文件访问
images_dir = Path(settings.storage_root) / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
