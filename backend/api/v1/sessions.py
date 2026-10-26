import io
import uuid
from datetime import datetime
from typing import Annotated, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_session_or_404
from core import transfer
from db import crud
from db.models import Session
from db.schemas import (
    MessageResponse,
    SessionCreate,
    SessionImportResult,
    SessionResponse,
    SessionUpdate,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    # 验证 KB 存在
    kb = await crud.get_kb(db, body.kb_id)
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base '{body.kb_id}' not found",
        )

    session = await crud.create_session(db, {
        "id": str(uuid.uuid4()),
        "kb_id": body.kb_id,
        "title": body.title,
        "llm_provider": body.llm_provider,
        "llm_model_id": body.llm_model_id,
        "llm_config": body.llm_config,
    })
    return session


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    kb_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await crud.list_sessions(db, kb_id)


@router.post("/import", response_model=SessionImportResult, status_code=status.HTTP_201_CREATED)
async def import_session(
    kb_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
):
    """把导出的会话归档导入到指定知识库。"""
    archive = await file.read()
    try:
        session = await transfer.import_session(db, archive, kb_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"会话导入失败：{e}",
        )

    return SessionImportResult(
        id=session.id,
        kb_id=session.kb_id,
        title=session.title,
        messages=await crud.count_messages(db, session.id),
    )


@router.get("/{session_id}/export")
async def export_session(
    session: Annotated[Session, Depends(get_session_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """导出会话：标题、模型配置、全部消息与消息附图。"""
    archive = await transfer.export_session(db, session)
    stem = session.title or f"session-{session.id[:8]}"
    filename = f"{transfer.safe_filename(stem, 'session')}.session.zip"
    return StreamingResponse(
        io.BytesIO(archive),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "X-Archive-Size": str(len(archive)),
        },
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session: Annotated[Session, Depends(get_session_or_404)],
):
    return session


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session: Annotated[Session, Depends(get_session_or_404)],
    body: SessionUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    update_data = body.model_dump(exclude_none=True)
    return await crud.update_session(db, session, update_data)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session: Annotated[Session, Depends(get_session_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await crud.delete_session(db, session)


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session: Annotated[Session, Depends(get_session_or_404)],
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = 1,
    page_size: int = 50,
    latest: bool = False,
    before: Optional[datetime] = None,
):
    """latest=true 取最近 page_size 条；before=<ISO 时间> 取该时间之前的一页。"""
    offset = (page - 1) * page_size
    return await crud.get_messages(
        db,
        session.id,
        limit=page_size,
        offset=offset,
        latest=latest,
        before=before,
    )
