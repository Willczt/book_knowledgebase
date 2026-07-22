import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_kb_or_404, get_qdrant
from db import crud
from db.models import KnowledgeBase
from db.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseStats,
)
from vector_store.qdrant_store import QdrantVectorStore

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Base"])


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    kb_id = str(uuid.uuid4())
    collection_name = f"kb_{kb_id.replace('-', '')}"

    # 创建 Qdrant 集合
    try:
        await qdrant.create_collection(collection_name, body.embedding_vector_dim)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create Qdrant collection: {e}",
        )

    kb = await crud.create_kb(db, {
        "id": kb_id,
        "name": body.name,
        "description": body.description,
        "embedding_provider": body.embedding_provider,
        "embedding_model_id": body.embedding_model_id,
        "embedding_vector_dim": body.embedding_vector_dim,
        "embedding_is_multimodal": body.embedding_is_multimodal,
        "caption_provider": body.caption_provider,
        "caption_model_id": body.caption_model_id,
        "qdrant_collection_name": collection_name,
    })
    return kb


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(db: Annotated[AsyncSession, Depends(get_db)]):
    return await crud.list_kbs(db)


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
):
    return kb


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    await qdrant.delete_collection(kb.qdrant_collection_name)
    await crud.delete_kb(db, kb)


@router.post("/{kb_id}/clear", status_code=status.HTTP_200_OK)
async def clear_knowledge_base(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    """清空向量库（删除所有向量点），保留 KB 元数据和文档记录。"""
    await qdrant.clear_collection(kb.qdrant_collection_name, kb.embedding_vector_dim)

    # 重置所有文档的 chunk/image 计数和状态
    docs = await crud.list_documents(db, kb.id)
    for doc in docs:
        await crud.update_document(db, doc, {
            "chunk_count": 0,
            "image_count": 0,
            "status": "pending",
            "processing_started_at": None,
            "processing_completed_at": None,
            "error_message": None,
        })

    return {"message": f"Knowledge base '{kb.name}' cleared successfully", "kb_id": kb.id}


@router.get("/{kb_id}/stats", response_model=KnowledgeBaseStats)
async def get_kb_stats(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    stats = await crud.get_kb_stats(db, kb.id)
    stats["qdrant_points_count"] = await qdrant.count_points(kb.qdrant_collection_name)
    return KnowledgeBaseStats(**stats)
