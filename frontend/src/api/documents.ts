import client from './client'
import type { Document, DocumentStatus } from '../types/api'

export const documentsApi = {
  upload: (kbId: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return client
      .post<Document>(`/knowledge-bases/${kbId}/documents`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
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
}
