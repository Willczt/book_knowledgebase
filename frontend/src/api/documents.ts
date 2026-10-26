import client from './client'
import type { Document, DocumentStatus } from '../types/api'

export const documentsApi = {
  upload: (kbId: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    // 不要手动设置 Content-Type：需要由浏览器补上 multipart 的 boundary
    return client
      .post<Document>(`/knowledge-bases/${kbId}/documents`, formData, {
        timeout: 300_000,
      })
      .then((r) => r.data)
  },

  list: (kbId: string) =>
    client.get<Document[]>(`/knowledge-bases/${kbId}/documents`).then((r) => r.data),

  status: (kbId: string, docId: string) =>
    client
      .get<DocumentStatus>(`/knowledge-bases/${kbId}/documents/${docId}/status`)
      .then((r) => r.data),

  delete: (kbId: string, docId: string) =>
    client.delete(`/knowledge-bases/${kbId}/documents/${docId}`),

  // 重建索引：清空向量库或摄入失败后使用
  reindex: (kbId: string, docId: string) =>
    client
      .post<Document>(`/knowledge-bases/${kbId}/documents/${docId}/reindex`)
      .then((r) => r.data),
}
