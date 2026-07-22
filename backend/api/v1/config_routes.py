from fastapi import APIRouter

from config import get_settings
from core.embedding.registry import AVAILABLE_EMBEDDING_MODELS
from core.llm.registry import get_available_llm_providers
from db.schemas import EmbeddingModelInfo, LLMProviderInfo

router = APIRouter(tags=["Config"])
settings = get_settings()


@router.get("/embedding-models", response_model=list[EmbeddingModelInfo])
async def list_embedding_models():
    """列出所有已配置的可用 embedding 模型（用于创建知识库时选择）"""
    available = []
    for m in AVAILABLE_EMBEDDING_MODELS:
        available.append(EmbeddingModelInfo(
            provider=m.provider,
            model_id=m.model_id,
            vector_dim=m.vector_dim,
            supports_multimodal=m.supports_multimodal,
            description=m.description,
        ))
    return available


@router.get("/llm-providers", response_model=list[LLMProviderInfo])
async def list_llm_providers():
    """列出所有已配置的可用 LLM 提供商"""
    return get_available_llm_providers()


@router.get("/health")
async def health_check():
    """检查数据库和 Qdrant 连接"""
    from db.engine import engine
    from qdrant_client import AsyncQdrantClient

    status = {"db": "ok", "qdrant": "ok"}

    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as e:
        status["db"] = f"error: {e}"

    try:
        client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        await client.get_collections()
        await client.close()
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    overall = "ok" if all(v == "ok" for v in status.values()) else "degraded"
    return {"status": overall, **status}
