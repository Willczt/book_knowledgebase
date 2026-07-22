from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ========================
# KnowledgeBase
# ========================

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    embedding_provider: str = Field(..., description="doubao | ollama | vllm")
    embedding_model_id: str
    embedding_vector_dim: int = Field(..., gt=0)
    embedding_is_multimodal: bool = False
    caption_provider: Optional[str] = None
    caption_model_id: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str]
    embedding_provider: str
    embedding_model_id: str
    embedding_vector_dim: int
    embedding_is_multimodal: bool
    caption_provider: Optional[str]
    caption_model_id: Optional[str]
    qdrant_collection_name: str
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseStats(BaseModel):
    total_docs: int
    indexed_docs: int
    failed_docs: int
    total_chunks: int
    total_images: int
    qdrant_points_count: int


# ========================
# Document
# ========================

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size_bytes: Optional[int]
    status: str
    error_message: Optional[str]
    chunk_count: int
    image_count: int
    doc_metadata: Optional[dict]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    id: str
    status: str
    chunk_count: int
    image_count: int
    error_message: Optional[str]


# ========================
# Session
# ========================

class SessionCreate(BaseModel):
    kb_id: str
    llm_provider: str = Field(..., description="doubao | deepseek | ollama | vllm")
    llm_model_id: str
    llm_config: Optional[dict] = None
    title: Optional[str] = None


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model_id: Optional[str] = None
    llm_config: Optional[dict] = None


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kb_id: str
    title: Optional[str]
    llm_provider: str
    llm_model_id: str
    llm_config: Optional[dict]
    created_at: datetime
    updated_at: datetime


# ========================
# Message
# ========================

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    image_paths: Optional[list]
    sources: Optional[list]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    created_at: datetime


# ========================
# Chat
# ========================

class RetrievalConfig(BaseModel):
    top_k: int = Field(default=8, ge=1, le=50)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_reranker: bool = False


class ChatRequest(BaseModel):
    content: str = Field(..., min_length=1)
    image: Optional[str] = None  # base64 编码的图片
    retrieval_config: Optional[RetrievalConfig] = None


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_doc: str
    doc_id: str
    page_num: Optional[int]
    chunk_type: str
    image_path: Optional[str]


class ChatResponse(BaseModel):
    message_id: str
    content: str
    sources: list[SourceChunk]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


# ========================
# Search
# ========================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    image: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=50)
    use_reranker: bool = False


class SearchResponse(BaseModel):
    results: list[SourceChunk]
    query: str


# ========================
# Config / Models
# ========================

class EmbeddingModelInfo(BaseModel):
    provider: str
    model_id: str
    vector_dim: int
    supports_multimodal: bool
    description: str


class LLMProviderInfo(BaseModel):
    provider: str
    model_id: str
    supports_vision: bool
    description: str
