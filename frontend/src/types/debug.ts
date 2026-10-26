// 调试模式（RAG 全链路 trace）相关类型，与后端 /api/v1/debug 对应

export interface DebugLayerInfo {
  name: string
  label: string
}

export interface DebugStatus {
  enabled: boolean
  layers: string[]
  available_layers: DebugLayerInfo[]
  defaults: { enabled: boolean; layers: string[] }
  max_traces: number
  persist: boolean
  trace_count: number
}

export interface TraceSummary {
  trace_id: string
  kind: string
  query: string
  started_at: string
  total_ms: number
  layers: string[]
  candidates: number
  reranked: boolean
  errors: number
}

export interface TraceStage {
  name: string
  duration_ms: number
  status: string
  error?: string | null
}

export interface TraceError {
  stage: string
  error: string
  at: string
}

export interface QuerySection {
  original?: string
  rewritten?: string
  rewrite?: { method: string; changed: boolean; mode?: string; reason?: string }
  retrieval_strategy?: string
  history_turns?: number
  has_image?: boolean
  kb_id?: string
  kb_name?: string
  session_id?: string
  mode?: string
  llm?: string
  rerank_backend?: string | null
  rerank_model?: string | null
  params?: Record<string, unknown>
  embedding?: {
    provider?: string
    model?: string | null
    dim?: number
    head?: number[]
    norm?: number
    sparse_terms?: number
    sparse_top_terms?: { index: number; weight: number }[]
    has_image?: boolean
  }
}

export interface RetrievalRow {
  chunk_id: string
  text: string
  text_preview?: string
  score: number
  rerank_score?: number | null
  vector_score?: number | null
  vector_rank?: number | null
  sparse_score?: number | null
  sparse_rank?: number | null
  source_doc: string
  doc_id: string
  page_num?: number | null
  chunk_type: string
  image_path?: string | null
  metadata?: Record<string, unknown>
  included: boolean
  drop_reason?: string | null
  final_rank?: number | null
}

export interface RetrievalSection {
  collection?: string
  params?: Record<string, unknown>
  candidate_count?: number
  filtered_count?: number
  threshold?: number
  returned?: number
  fusion?: { method?: string; k?: number; prefetch_limit?: number } | null
  cache?: { enabled: boolean; hit: boolean; note?: string }
  vector_hits?: number
  sparse_hits?: number
  warnings?: string[]
  errors?: Record<string, string>
  results?: RetrievalRow[]
}

export interface RerankSection {
  enabled?: boolean
  /** applied | skipped | disabled | unavailable | cooldown | failed */
  status?: string
  /** 未执行重排时的原因说明 */
  reason?: string
  backend?: string
  model?: string
  candidates?: number
  duration_ms?: number
  top_score?: number | null
  order_before?: string[]
  order_after?: string[]
  scores?: Record<string, number>
  dropped?: string[]
  error?: string
  candidates_detail?: {
    chunk_id: string
    rank_before: number
    rank_after: number
    score_before: number
    score_after: number
  }[]
}

export interface GenerationSection {
  params?: Record<string, unknown>
  prompt?: {
    system?: string
    messages?: { role: string; content: string }[]
    context_text?: string
    context_ref_count?: number
  }
  context_chunks?: {
    ref: number
    chunk_id: string
    source_doc: string
    page_num?: number | null
    chunk_type: string
    tokens: number
    order: number
    included: boolean
    truncated: boolean
    text: string
  }[]
  truncation?: { applied: boolean; strategy: string; dropped: string[]; note?: string }
  tokens?: {
    prompt_estimated?: number
    context_estimated?: number
    completion_estimated?: number
    estimated?: boolean
    note?: string
  }
  output?: string
  error?: string
  citations?: { ref: number; chunk_id: string; cited: boolean; evidence?: string | null }[]
}

export interface TraceQuality {
  k: number
  relevant: number
  hits?: number
  recall?: number
  precision?: number
  mrr?: number
  ndcg?: number
  citation_coverage?: number
  ranked_ids?: string[]
  note?: string | null
}

export interface TraceDetail {
  trace_id: string
  kind: string
  /** 原始问题（新格式），Query 段对象见 query */
  query_text?: string
  query?: string | QuerySection
  started_at: string
  total_ms: number
  layers: string[]
  errors: TraceError[]
  stages?: TraceStage[]
  retrieval?: RetrievalSection
  rerank?: RerankSection
  generation?: GenerationSection
  label?: { verdict: 'good' | 'bad'; stage?: string | null; note?: string | null }
  quality?: TraceQuality
  [section: string]: unknown
}

export interface StageMetric {
  stage: string
  count: number
  avg_ms: number
  p50_ms: number
  p95_ms: number
  max_ms: number
  errors: number
}

export interface SearchDebugRequest {
  query: string
  top_k: number
  score_threshold: number
  use_reranker: boolean
  rerank_candidates: number
}

export interface SearchDebugResponse {
  query: string
  trace_id?: string | null
  results: { chunk_id: string; score: number; source_doc: string; text: string }[]
}
