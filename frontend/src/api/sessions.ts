import client from './client'
import type { Session, SessionCreate, SessionImportResult, Message } from '../types/api'

type SessionUpdateData = Partial<Omit<SessionCreate, 'kb_id'>>

export const sessionsApi = {
  create: (data: SessionCreate) =>
    client.post<Session>('/sessions', data).then((r) => r.data),

  list: (kbId: string) =>
    client.get<Session[]>('/sessions', { params: { kb_id: kbId } }).then((r) => r.data),

  get: (id: string) =>
    client.get<Session>(`/sessions/${id}`).then((r) => r.data),

  update: (id: string, data: SessionUpdateData) =>
    client.patch<Session>(`/sessions/${id}`, data).then((r) => r.data),

  delete: (id: string) => client.delete(`/sessions/${id}`),

  // latest=true 取最近 N 条（长会话不会只看到最早的那批）
  messages: (id: string, page = 1, pageSize = 50, latest = false) =>
    client
      .get<Message[]>(`/sessions/${id}/messages`, {
        params: { page, page_size: pageSize, latest },
      })
      .then((r) => r.data),

  // 取 before 之前的最近 N 条（升序），用于「加载更早」
  history: (id: string, before: string, pageSize = 50) =>
    client
      .get<Message[]>(`/sessions/${id}/messages`, {
        params: { before, page_size: pageSize },
      })
      .then((r) => r.data),

  exportArchive: (id: string) =>
    client
      .get<Blob>(`/sessions/${id}/export`, { responseType: 'blob', timeout: 0 })
      .then((r) => r.data),

  importArchive: (kbId: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return client
      .post<SessionImportResult>('/sessions/import', form, {
        params: { kb_id: kbId },
        timeout: 0,
      })
      .then((r) => r.data)
  },
}
