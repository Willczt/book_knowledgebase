from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Document, KnowledgeBase, Message, Session


# ========================
# KnowledgeBase CRUD
# ========================

async def create_kb(db: AsyncSession, data: dict) -> KnowledgeBase:
    kb = KnowledgeBase(**data)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


async def get_kb(db: AsyncSession, kb_id: str) -> Optional[KnowledgeBase]:
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    return result.scalar_one_or_none()


async def list_kbs(db: AsyncSession) -> list[KnowledgeBase]:
    result = await db.execute(select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()))
    return list(result.scalars().all())


async def delete_kb(db: AsyncSession, kb: KnowledgeBase) -> None:
    await db.delete(kb)
    await db.commit()


# ========================
# Document CRUD
# ========================

async def create_document(db: AsyncSession, data: dict) -> Document:
    doc = Document(**data)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def get_document(db: AsyncSession, doc_id: str) -> Optional[Document]:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def list_documents(
    db: AsyncSession,
    kb_id: str,
    status: Optional[str] = None,
) -> list[Document]:
    q = select(Document).where(Document.kb_id == kb_id)
    if status:
        q = q.where(Document.status == status)
    q = q.order_by(Document.created_at.desc())
    result = await db.execute(q)
    return list(result.scalars().all())


async def update_document(db: AsyncSession, doc: Document, data: dict) -> Document:
    for key, value in data.items():
        setattr(doc, key, value)
    await db.commit()
    await db.refresh(doc)
    return doc


async def delete_document(db: AsyncSession, doc: Document) -> None:
    await db.delete(doc)
    await db.commit()


async def list_unfinished_documents(db: AsyncSession) -> list[Document]:
    """列出所有还没摄入完成的文档（pending / processing）"""
    result = await db.execute(
        select(Document).where(Document.status.in_(("pending", "processing")))
    )
    return list(result.scalars().all())


async def fail_documents(db: AsyncSession, doc_ids: list[str], message: str) -> int:
    """把指定文档标记为 failed（用于清理僵尸摄入任务）"""
    if not doc_ids:
        return 0
    await db.execute(
        update(Document)
        .where(Document.id.in_(doc_ids))
        .values(
            status="failed",
            error_message=message,
            processing_completed_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
    return len(doc_ids)


async def get_kb_stats(db: AsyncSession, kb_id: str) -> dict:
    docs = await list_documents(db, kb_id)
    total_docs = len(docs)
    indexed_docs = sum(1 for d in docs if d.status == "indexed")
    failed_docs = sum(1 for d in docs if d.status == "failed")
    total_chunks = sum(d.chunk_count for d in docs)
    total_images = sum(d.image_count for d in docs)
    return {
        "total_docs": total_docs,
        "indexed_docs": indexed_docs,
        "failed_docs": failed_docs,
        "total_chunks": total_chunks,
        "total_images": total_images,
    }


# ========================
# Session CRUD
# ========================

async def create_session(db: AsyncSession, data: dict) -> Session:
    session = Session(**data)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session_by_id(db: AsyncSession, session_id: str) -> Optional[Session]:
    result = await db.execute(select(Session).where(Session.id == session_id))
    return result.scalar_one_or_none()


async def list_sessions(db: AsyncSession, kb_id: str) -> list[Session]:
    result = await db.execute(
        select(Session)
        .where(Session.kb_id == kb_id)
        .order_by(Session.updated_at.desc())
    )
    return list(result.scalars().all())


async def update_session(db: AsyncSession, session: Session, data: dict) -> Session:
    for key, value in data.items():
        if value is not None:
            setattr(session, key, value)
    await db.commit()
    await db.refresh(session)
    return session


async def delete_session(db: AsyncSession, session: Session) -> None:
    await db.delete(session)
    await db.commit()


# ========================
# Message CRUD
# ========================

async def create_message(db: AsyncSession, data: dict) -> Message:
    message = Message(**data)
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def get_messages(
    db: AsyncSession,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def get_recent_messages(db: AsyncSession, session_id: str, turns: int = 10) -> list[Message]:
    """获取最近 N 条消息（按时间升序排列，用于构建对话历史）"""
    # created_at 由数据库生成，精度只到秒；同一轮 user/assistant 常落在同一秒。
    # 用 role 做兜底排序，保证同一时刻 user 一定排在 assistant 之前，否则历史顺序会错乱。
    role_order = case((Message.role == "user", 0), else_=1)
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc(), role_order.desc())
        .limit(turns * 2)  # 每轮 user+assistant 各一条
    )
    messages = list(result.scalars().all())
    return list(reversed(messages))


async def count_messages(db: AsyncSession, session_id: str) -> int:
    result = await db.execute(
        select(func.count()).where(Message.session_id == session_id)
    )
    return result.scalar_one()
