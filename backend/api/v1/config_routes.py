from fastapi import APIRouter

from config import get_settings
from core.embedding.registry import get_available_embedding_models
from core.llm.registry import get_available_llm_providers
from db.schemas import EmbeddingModelInfo, LLMProviderInfo

router = APIRouter(tags=["Config"])
settings = get_settings()


@router.get("/embedding-models", response_model=list[EmbeddingModelInfo])
async def list_embedding_models():
    """列出所有已配置的可用 embedding 模型（用于创建知识库时选择）"""
    return [
        EmbeddingModelInfo(
            provider=m.provider,
            model_id=m.model_id,
            vector_dim=m.vector_dim,
            supports_multimodal=m.supports_multimodal,
            description=m.description,
        )
        for m in get_available_embedding_models()
    ]


@router.get("/llm-providers", response_model=list[LLMProviderInfo])
async def list_llm_providers():
    """列出所有已配置的可用 LLM 提供商"""
    return get_available_llm_providers()


@router.get("/health")
async def health_check():
    """
    检查数据库 / Qdrant 连接，并探测外部模型服务（Ollama、DeepSeek、豆包）的连通性。

    providers 里每项的 ok=false 表示该服务当前不可用，error 字段给出可读原因，
    前端可据此提前提示"embedding 后端未启动"，而不用等到摄入失败。
    """
    import asyncio

    from sqlalchemy import text

    from api.deps import get_qdrant
    from core import health
    from db.engine import engine

    status = {"db": "ok", "qdrant": "ok"}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        status["db"] = f"error: {e}"

    # 复用共享客户端：健康检查每 30s 一次，新建客户端会持续泄漏连接池
    try:
        await get_qdrant().client.get_collections()
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    # 三个外部服务并发探测，任一失败不影响其它
    ollama_result, deepseek_result, ark_result = await asyncio.gather(
        health.check_ollama(),
        health.check_deepseek(),
        health.check_ark(),
    )

    providers = {
        "ollama": ollama_result,
        "deepseek": deepseek_result,
        "ark": ark_result,
    }

    overall = "ok" if all(v == "ok" for v in status.values()) else "degraded"
    return {"status": overall, **status, "providers": providers}
