import client from './client'
import type {
  ExportJobStatus,
  KnowledgeBase,
  KnowledgeBaseCreate,
  KnowledgeBaseImportResult,
  KnowledgeBaseStats,
  KnowledgeBaseUpdate,
} from '../types/api'

export const knowledgeBaseApi = {
  list: () => client.get<KnowledgeBase[]>('/knowledge-bases').then((r) => r.data),

  create: (data: KnowledgeBaseCreate) =>
    client.post<KnowledgeBase>('/knowledge-bases', data).then((r) => r.data),

  get: (id: string) =>
    client.get<KnowledgeBase>(`/knowledge-bases/${id}`).then((r) => r.data),

  update: (id: string, data: KnowledgeBaseUpdate) =>
    client.patch<KnowledgeBase>(`/knowledge-bases/${id}`, data).then((r) => r.data),

  delete: (id: string) => client.delete(`/knowledge-bases/${id}`),

  clear: (id: string) =>
    client.post(`/knowledge-bases/${id}/clear`).then((r) => r.data),

  stats: (id: string) =>
    client.get<KnowledgeBaseStats>(`/knowledge-bases/${id}/stats`).then((r) => r.data),

  // 导出包含文档、图片、向量库与历史会话，体积可能较大
  exportArchive: (id: string) =>
    client
      .get<Blob>(`/knowledge-bases/${id}/export`, {
        responseType: 'blob',
        timeout: 0,
      })
      .then((r) => r.data),

  // 后台导出：立即返回 job，进度靠轮询（大库导出可能几十秒以上）
  startExportJob: (id: string) =>
    client.post<ExportJobStatus>(`/knowledge-bases/${id}/export/jobs`).then((r) => r.data),

  exportJobStatus: (id: string, jobId: string) =>
    client
      .get<ExportJobStatus>(`/knowledge-bases/${id}/export/jobs/${jobId}`)
      .then((r) => r.data),

  downloadExportJob: (id: string, jobId: string) =>
    client
      .get<Blob>(`/knowledge-bases/${id}/export/jobs/${jobId}/download`, {
        responseType: 'blob',
        timeout: 0,
      })
      .then((r) => r.data),

  importArchive: (file: File, name?: string) => {
    const form = new FormData()
    form.append('file', file)
    return client
      .post<KnowledgeBaseImportResult>('/knowledge-bases/import', form, {
        params: name ? { name } : undefined,
        timeout: 0,
      })
      .then((r) => r.data)
  },
}
