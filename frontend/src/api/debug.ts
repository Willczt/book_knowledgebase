import client from './client'
import type {
  DebugStatus,
  SearchDebugRequest,
  SearchDebugResponse,
  StageMetric,
  TraceDetail,
  TraceQuality,
  TraceSummary,
} from '../types/debug'

export const debugApi = {
  status: () => client.get<DebugStatus>('/debug/status').then((r) => r.data),

  updateStatus: (body: { enabled?: boolean; layers?: string[] }) =>
    client.put<DebugStatus>('/debug/status', body).then((r) => r.data),

  traces: (limit = 50) =>
    client.get<TraceSummary[]>('/debug/traces', { params: { limit } }).then((r) => r.data),

  trace: (traceId: string) =>
    client.get<TraceDetail>(`/debug/traces/${traceId}`).then((r) => r.data),

  clear: () => client.delete<{ removed: number }>('/debug/traces').then((r) => r.data),

  metrics: (limit = 100) =>
    client
      .get<{ traces: number; stages: StageMetric[] }>('/debug/metrics', { params: { limit } })
      .then((r) => r.data),

  label: (traceId: string, body: { verdict: 'good' | 'bad'; stage?: string; note?: string }) =>
    client.post(`/debug/traces/${traceId}/label`, body).then((r) => r.data),

  evaluate: (traceId: string, body: { relevant_chunk_ids: string[]; k?: number }) =>
    client
      .post<{ trace_id: string; quality: TraceQuality }>(`/debug/traces/${traceId}/evaluate`, body)
      .then((r) => r.data),

  /** 参数热调：用指定参数重跑一次检索，返回新的 trace_id 便于对比 */
  search: (kbId: string, body: SearchDebugRequest) =>
    client
      .post<SearchDebugResponse>(`/knowledge-bases/${kbId}/search`, body, { timeout: 120_000 })
      .then((r) => r.data),
}
