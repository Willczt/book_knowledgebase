import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """毫秒级 UTC 时间（naive，与 SQLite CURRENT_TIMESTAMP 保持同一时区语义）。

    消息顺序依赖 created_at：数据库默认值只到秒，同一秒内的问答对会排错，
    所以这里用 Python 生成带微秒的时间戳。
    """
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class FileType(str, Enum):
    pdf = "pdf"
    epub = "epub"
    markdown = "markdown"


class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Embedding 锁定字段（创建后不可变）
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_vector_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_is_multimodal: Mapped[bool] = mapped_column(Boolean, default=False)

    # Caption 模型配置
    caption_provider: Mapped[Optional[str]] = mapped_column(String(64))
    caption_model_id: Mapped[Optional[str]] = mapped_column(String(128))

    qdrant_collection_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["Session"]] = relationship(
        "Session", back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kb_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    doc_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # 任务队列字段：DB 即队列，进程重启/多 worker 都能靠租约恢复
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        Index("ix_documents_queue", "status", "next_retry_at"),
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship("KnowledgeBase", back_populates="documents")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kb_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(512))
    llm_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    llm_config: Mapped[Optional[dict]] = mapped_column(JSON)  # temperature, max_tokens 等
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    knowledge_base: Mapped[KnowledgeBase] = relationship("KnowledgeBase", back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    image_paths: Mapped[Optional[list]] = mapped_column(JSON)  # 用户消息附带的图片路径
    sources: Mapped[Optional[list]] = mapped_column(JSON)       # 助手消息的检索来源
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    session: Mapped[Session] = relationship("Session", back_populates="messages")
