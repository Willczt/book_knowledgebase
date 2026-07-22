import base64
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.crud import get_kb, get_session_by_id
from db.engine import get_session
from db.models import KnowledgeBase, Session
from core.embedding.base import EmbeddingProvider
from core.embedding.registry import get_embedding_registry, EmbeddingRegistry
from core.exceptions import EmbeddingModelMismatch
from vector_store.qdrant_store import QdrantVectorStore


async def get_db() -> AsyncSession:
    async for session in get_session():
        yield session


def get_qdrant() -> QdrantVectorStore:
    return QdrantVectorStore()


def get_registry() -> EmbeddingRegistry:
    return get_embedding_registry()


async def get_kb_or_404(
    kb_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KnowledgeBase:
    kb = await get_kb(db, kb_id)
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{kb_id}' not found",
        )
    return kb


async def get_session_or_404(
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Session:
    session = await get_session_by_id(db, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return session


async def validate_kb_embedding(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    registry: Annotated[EmbeddingRegistry, Depends(get_registry)],
) -> EmbeddingProvider:
    """
    关键守卫：确保 KB 锁定的 embedding 模型当前可用且维度匹配。
    如果不一致，立即返回 409 Conflict。
    """
    try:
        # 尝试动态注册（如果不在 registry 中）
        try:
            provider = registry.get(kb.embedding_provider, kb.embedding_model_id)
        except KeyError:
            provider = registry.register_dynamic(
                kb.embedding_provider,
                kb.embedding_model_id,
                kb.embedding_vector_dim,
            )

        if provider.model_info.vector_dim != kb.embedding_vector_dim:
            raise EmbeddingModelMismatch(
                f"Vector dimension mismatch: KB expects {kb.embedding_vector_dim}, "
                f"model provides {provider.model_info.vector_dim}"
            )
        return provider

    except EmbeddingModelMismatch as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Embedding provider '{kb.embedding_provider}' with model "
                f"'{kb.embedding_model_id}' is not currently available: {e}. "
                "Ensure the provider is running or reconfigure the knowledge base."
            ),
        )


def decode_base64_image(b64_str: Optional[str]) -> Optional[bytes]:
    if not b64_str:
        return None
    # 支持 data:image/...;base64,... 格式
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        return base64.b64decode(b64_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data",
        )
