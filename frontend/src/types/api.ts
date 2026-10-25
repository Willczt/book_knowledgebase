// API 类型定义（与后端 Pydantic schema 对应）

export interface KnowledgeBase {
  id: string
  name: string
  description?: string
  embedding_provider: string
  embedding_model_id: string
  embedding_vector_dim: number
  embedding_is_multimodal: boolean
  caption_provider?: string
  caption_model_id?: string
  qdrant_collection_name: string
  created_at: string
  updated_at: string
}

export interface KnowledgeBaseCreate {
  name: string
  description?: string
  embedding_provider: string
  embedding_model_id: string
  embedding_vector_dim: number
  embedding_is_multimodal?: boolean
  caption_provider?: string
  caption_model_id?: string
}

export interface KnowledgeBaseStats {
  total_docs: number
  indexed_docs: number
  failed_docs: number
  total_chunks: number
  total_images: number
  qdrant_points_count: number
}

export interface Document {
  id: string
  kb_id: string
  filename: string
  file_type: string
  file_size_bytes?: number
  status: 'pending' | 'processing' | 'indexed' | 'failed'
  error_message?: string
  chunk_count: number
  image_count: number
  doc_metadata?: Record<string, unknown>
  processing_started_at?: string
  processing_completed_at?: string
  created_at: string
}

export interface IngestProgress {
  stage: string
  stage_label: string
  percent: number
  current?: number | null
  total?: number | null
  message?: string | null
}

export interface DocumentStatus {
  id: string
  status: string
  chunk_count: number
  image_count: number
  error_message?: string
  progress?: IngestProgress | null
}

export interface ProviderHealth {
  ok: boolean
  url?: string
  configured?: boolean
  latency_ms?: number
  error?: string
  model?: string
  model_count?: number
  has_embedding_model?: boolean
  has_vision_model?: boolean
  model_exists?: boolean
  supports_vision?: boolean
  note?: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  db: string
  qdrant: string
  providers: {
    ollama: ProviderHealth
    deepseek: ProviderHealth
    ark: ProviderHealth
  }
}

export interface Session {
  id: string
  kb_id: string
  title?: string
  llm_provider: string
  llm_model_id: string
  llm_config?: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface SessionCreate {
  kb_id: string
  llm_provider: string
  llm_model_id: string
  llm_config?: Record<string, unknown>
  title?: string
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  image_paths?: string[]
  sources?: SourceChunk[]
  prompt_tokens?: number
  completion_tokens?: number
  created_at: string
}

export interface SourceChunk {
  chunk_id: string
  text: string
  score: number
  source_doc: string
  doc_id: string
  page_num?: number
  chunk_type: 'text' | 'image_caption'
  image_path?: string
}

export interface ChatRequest {
  content: string
  image?: string
  retrieval_config?: {
    top_k?: number
    score_threshold?: number
    use_reranker?: boolean
  }
}

export interface EmbeddingModelInfo {
  provider: string
  model_id: string
  vector_dim: number
  supports_multimodal: boolean
  description: string
}

export interface LLMProviderInfo {
  provider: string
  model_id: string
  supports_vision: boolean
  description: string
}

// SSE 事件类型
export interface SSETokenEvent {
  token: string
}

export interface SSEDoneEvent {
  done: true
  message_id: string
  sources: SourceChunk[]
}

export interface SSEErrorEvent {
  error: string
}

export type SSEEvent = SSETokenEvent | SSEDoneEvent | SSEErrorEvent
