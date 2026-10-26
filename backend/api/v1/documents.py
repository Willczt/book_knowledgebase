import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_kb_or_404, get_qdrant
from config import get_settings
from db import crud
from db.models import KnowledgeBase
from db.schemas import DocumentResponse, DocumentStatusResponse
from tasks import progress as ingest_progress
from tasks.background import spawn
from tasks.ingest_task import run_ingest_task
from vector_store.qdrant_store import QdrantVectorStore

router = APIRouter(prefix="/knowledge-bases", tags=["Documents"])
settings = get_settings()
logger = structlog.get_logger()

ALLOWED_EXTENSIONS = {".pdf", ".epub", ".md", ".markdown", ".txt", ".text"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


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

    # 分块落盘：避免 100MB 的文件被整块读进内存
    doc_id = str(uuid.uuid4())
    file_type = suffix.lstrip(".")
    storage_dir = Path(settings.storage_root) / "docs" / kb.id
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / f"{doc_id}{suffix}"

    size = 0
    try:
        with file_path.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds max allowed {MAX_FILE_SIZE}",
                    )
                out.write(chunk)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    # 创建文档记录
    doc = await crud.create_document(db, {
        "id": doc_id,
        "kb_id": kb.id,
        "filename": file.filename,
        "file_type": file_type,
        "file_path": str(file_path),
        "file_size_bytes": size,
        "status": "pending",
    })

    # 启动后台摄入任务（经 spawn 持有引用，避免任务被 GC 回收）
    spawn(
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


@router.post(
    "/{kb_id}/documents/{doc_id}/reindex",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_document(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    doc_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """用已保存的原始文件重建索引（清空向量库、或上次摄入失败后使用）。"""
    doc = await crud.get_document(db, doc_id)
    if doc is None or doc.kb_id != kb.id:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status == "processing" or ingest_progress.get(doc.id):
        raise HTTPException(status_code=409, detail="文档正在处理中，请稍后再试")
    if not Path(doc.file_path).is_file():
        raise HTTPException(status_code=409, detail="原始文件已丢失，请删除后重新上传")

    doc = await crud.update_document(db, doc, {
        "status": "pending",
        "error_message": None,
        "chunk_count": 0,
        "image_count": 0,
        "processing_started_at": None,
        "processing_completed_at": None,
    })

    spawn(
        run_ingest_task(
            doc_id=doc.id,
            file_path=doc.file_path,
            source_doc=doc.filename,
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

    # 摄入进度只存在于进程内存，仅处理中才有意义
    progress_data = None
    if doc.status in ("pending", "processing"):
        progress_data = ingest_progress.get(doc_id)

    return DocumentStatusResponse(
        id=doc.id,
        status=doc.status,
        chunk_count=doc.chunk_count,
        image_count=doc.image_count,
        error_message=doc.error_message,
        progress=progress_data,
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

    # 删除存储的文件（删除失败不应阻断数据库清理，但要留痕）
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception as e:
        logger.warning("delete_document_file_failed", doc_id=doc_id, error=str(e))

    # 删除该文档提取出的图片，避免磁盘泄漏
    try:
        image_dir = Path(settings.storage_root) / "images" / kb.id / doc_id
        if image_dir.exists():
            shutil.rmtree(image_dir, ignore_errors=True)
    except Exception as e:
        logger.warning("delete_document_images_failed", doc_id=doc_id, error=str(e))

    await crud.delete_document(db, doc)
