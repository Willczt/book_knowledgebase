import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from api.deps import get_db, get_kb_or_404, get_qdrant
from core import export_jobs, transfer
from db import crud
from db.engine import AsyncSessionLocal
from db.models import KnowledgeBase
from db.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseImportResult,
    KnowledgeBaseResponse,
    KnowledgeBaseStats,
    KnowledgeBaseUpdate,
)
from vector_store.qdrant_store import QdrantVectorStore

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Base"])

# 后台导出任务需要持有引用，否则可能被 GC 回收
_export_tasks: set[asyncio.Task] = set()


def _remove_file(path: Path) -> None:
    """响应发送完毕后清理临时导出文件"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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

    try:
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
    except Exception:
        # 入库失败时回滚已建的集合，避免留下孤儿 collection
        await db.rollback()
        await qdrant.delete_collection(collection_name)
        raise
    return kb


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(db: Annotated[AsyncSession, Depends(get_db)]):
    return await crud.list_kbs(db)


@router.post("/import", response_model=KnowledgeBaseImportResult, status_code=status.HTTP_201_CREATED)
async def import_knowledge_base(
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
    file: UploadFile = File(...),
    name: Optional[str] = None,
):
    """导入知识库归档包（含文档、图片、向量库与历史会话）。"""
    # UploadFile 超过阈值会自动落到磁盘临时文件，直接交给 zipfile 读取，不整包进内存
    max_bytes = transfer.MAX_IMPORT_BYTES
    if getattr(file, "size", None) and file.size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="导入包超过 2GB，无法导入",
        )

    try:
        result = await transfer.import_kb(db, qdrant, file.file, name=name)
    except transfer.KnowledgeBaseExists as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导入失败：{e}",
        )

    kb = result["kb"]
    return KnowledgeBaseImportResult(
        id=kb.id,
        name=kb.name,
        documents=result["documents"],
        sessions=result["sessions"],
        messages=result["messages"],
        vectors=result["vectors"],
    )


@router.patch("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    body: KnowledgeBaseUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """重命名 / 修改描述。embedding 等锁定字段不可改。"""
    data = body.model_dump(exclude_unset=True)
    if not data:
        return kb
    if "name" in data:
        name = (data["name"] or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="知识库名称不能为空",
            )
        data["name"] = name
    return await crud.update_kb(db, kb, data)


@router.get("/{kb_id}/export")
async def export_knowledge_base(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    """导出知识库：元数据 + 文档 + 图片 + 向量库快照 + 历史会话。

    先写入临时文件再流式下发，向量库再大也不会撑爆内存。
    """
    filename = f"{transfer.safe_filename(kb.name, 'knowledge-base')}.kbz.zip"

    fd, tmp_name = tempfile.mkstemp(prefix="kb-export-", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        await transfer.export_kb(db, kb, qdrant, tmp_path)
    except Exception:
        _remove_file(tmp_path)
        raise

    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(_remove_file, tmp_path),
    )


async def _run_export_job(
    job: export_jobs.ExportJob,
    kb_id: str,
    qdrant: QdrantVectorStore,
) -> None:
    """后台线程里跑导出：自己开 DB 会话，进度写回任务状态。"""
    fd, tmp_name = tempfile.mkstemp(prefix="kb-export-", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)

    def on_progress(stage: str, percent: float) -> None:
        export_jobs.update(job, percent=percent, stage=stage)

    try:
        async with AsyncSessionLocal() as db:
            kb = await crud.get_kb(db, kb_id)
            if kb is None:
                raise RuntimeError("知识库已被删除")
            await transfer.export_kb(db, kb, qdrant, tmp_path, progress=on_progress)
        export_jobs.finish(job, tmp_path)
    except Exception as exc:
        _remove_file(tmp_path)
        export_jobs.fail(job, f"{type(exc).__name__}: {exc}")


@router.post("/{kb_id}/export/jobs", status_code=status.HTTP_202_ACCEPTED)
async def start_export_job(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    """启动后台导出任务，立即返回 job_id；进度用下面的状态接口轮询。"""
    running = export_jobs.find_running(kb.id)
    if running is not None:
        return running.to_dict()

    filename = f"{transfer.safe_filename(kb.name, 'knowledge-base')}.kbz.zip"
    job = export_jobs.create_job(kb.id, kb.name, filename)
    task = asyncio.create_task(_run_export_job(job, kb.id, qdrant))
    _export_tasks.add(task)
    task.add_done_callback(_export_tasks.discard)
    return job.to_dict()


@router.get("/{kb_id}/export/jobs/{job_id}")
async def export_job_status(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    job_id: str,
):
    job = export_jobs.get_job(job_id)
    if job is None or job.kb_id != kb.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="导出任务不存在或已过期")
    return job.to_dict()


@router.get("/{kb_id}/export/jobs/{job_id}/download")
async def download_export_job(
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    job_id: str,
):
    job = export_jobs.get_job(job_id)
    if job is None or job.kb_id != kb.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="导出任务不存在或已过期")
    if job.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=job.error or "导出失败",
        )
    if job.status != "done" or job.path is None or not job.path.is_file():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="导出尚未完成")

    return FileResponse(
        job.path,
        media_type="application/zip",
        filename=job.filename,
        background=BackgroundTask(export_jobs.mark_downloaded, job),
    )


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
