import client from './client'
import type { Session, SessionCreate, Message } from '../types/api'

export const sessionsApi = {
  create: (data: SessionCreate) =>
    client.post<Session>('/sessions', data).then((r) => r.data),

  list: (kbId: string) =>
    client.get<Session[]>('/sessions', { params: { kb_id: kbId } }).then((r) => r.data),

  get: (id: string) =>
    client.get<Session>(`/sessions/${id}`).then((r) => r.data),

  update: (id: string, data: Partial<SessionCreate>) =>
    client.patch<Session>(`/sessions/${id}`, data).then((r) => r.data),

  delete: (id: string) => client.delete(`/sessions/${id}`),

  messages: (id: string, page = 1, pageSize = 50) =>
    client
      .get<Message[]>(`/sessions/${id}/messages`, { params: { page, page_size: pageSize } })
      .then((r) => r.data),
}
