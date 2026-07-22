from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.v1.router import api_router
from config import get_settings
from db.engine import init_db

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动时初始化数据库
    await init_db()

    # 确保存储目录存在
    for subdir in ["docs", "images"]:
        Path(settings.storage_root, subdir).mkdir(parents=True, exist_ok=True)

    logger.info("application_started", storage_root=settings.storage_root)
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="电子书 RAG 知识库 API",
    description="支持 epub/pdf/markdown 的多模态 RAG 系统，混合检索，多模型支持",
    version="0.1.0",
    lifespan=lifespan,
)

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
