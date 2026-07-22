import asyncio
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_kb_or_404
from config import get_settings
from db import crud
from db.models import KnowledgeBase
from db.schemas import DocumentResponse, DocumentStatusResponse
from tasks.ingest_task import run_ingest_task
from vector_store.qdrant_store import QdrantVectorStore
from api.deps import get_qdrant

router = APIRouter(prefix="/knowledge-bases", tags=["Documents"])
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".epub", ".md", ".markdown"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


@router.post(
    "/{kb_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {len(content)} exceeds max allowed {MAX_FILE_SIZE}",
        )

    # 存储文件
    doc_id = str(uuid.uuid4())
    file_type = suffix.lstrip(".")
    if file_type == "markdown":
        file_type = "markdown"

    storage_dir = Path(settings.storage_root) / "docs" / kb.id
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / f"{doc_id}{suffix}"
    file_path.write_bytes(content)

    # 创建文档记录
    doc = await crud.create_document(db, {
        "id": doc_id,
        "kb_id": kb.id,
        "filename": file.filename,
        "file_type": file_type,
        "file_path": str(file_path),
        "file_size_bytes": len(content),
        "status": "pending",
    })

    # 启动后台摄入任务
    asyncio.create_task(
        run_ingest_task(
            doc_id=doc_id,
            file_path=str(file_path),
            source_doc=file.filename or doc_id,
            kb_id=kb.id,
            kb_embedding_provider=kb.embedding_provider,
            kb_embedding_model_id=kb.embedding_model_id,
            kb_embedding_vector_dim=kb.embedding_vector_dim,
            kb_qdrant_collection=kb.qdrant_collection_name,
            caption_provider=kb.caption_provider,
            caption_model_id=kb.caption_model_id,
        )
    )

    return doc


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    doc_status: str | None = None,
):
    return await crud.list_documents(db, kb.id, status=doc_status)


@router.get("/{kb_id}/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    doc_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    doc = await crud.get_document(db, doc_id)
    if doc is None or doc.kb_id != kb.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        id=doc.id,
        status=doc.status,
        chunk_count=doc.chunk_count,
        image_count=doc.image_count,
        error_message=doc.error_message,
    )


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    doc_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    doc = await crud.get_document(db, doc_id)
    if doc is None or doc.kb_id != kb.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    doc_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    doc = await crud.get_document(db, doc_id)
    if doc is None or doc.kb_id != kb.id:
        raise HTTPException(status_code=404, detail="Document not found")

    # 删除 Qdrant 向量
    await qdrant.delete_doc_points(kb.qdrant_collection_name, doc_id)

    # 删除存储的文件（可选：保留原文件，仅删元数据）
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception:
        pass

    await crud.delete_document(db, doc)
