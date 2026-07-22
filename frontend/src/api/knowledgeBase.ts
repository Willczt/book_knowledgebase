import client from './client'
import type { KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseStats } from '../types/api'

export const knowledgeBaseApi = {
  list: () => client.get<KnowledgeBase[]>('/knowledge-bases').then((r) => r.data),

  create: (data: KnowledgeBaseCreate) =>
    client.post<KnowledgeBase>('/knowledge-bases', data).then((r) => r.data),

  get: (id: string) =>
    client.get<KnowledgeBase>(`/knowledge-bases/${id}`).then((r) => r.data),

  delete: (id: string) => client.delete(`/knowledge-bases/${id}`),

  clear: (id: string) =>
    client.post(`/knowledge-bases/${id}/clear`).then((r) => r.data),

  stats: (id: string) =>
    client.get<KnowledgeBaseStats>(`/knowledge-bases/${id}/stats`).then((r) => r.data),
}
