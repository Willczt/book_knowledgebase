import { create } from 'zustand'
import type { Session, Message } from '../types/api'

interface SessionStore {
  sessions: Session[]
  activeSession: Session | null
  messages: Message[]
  setSessions: (sessions: Session[]) => void
  setActiveSession: (session: Session | null) => void
  addSession: (session: Session) => void
  updateSession: (session: Session) => void
  removeSession: (id: string) => void
  setMessages: (messages: Message[]) => void
  appendMessage: (message: Message) => void
  updateMessage: (id: string, patch: Partial<Message>) => void
}

export const useSessionStore = create<SessionStore>((set) => ({
  sessions: [],
  activeSession: null,
  messages: [],
  setSessions: (sessions) => set({ sessions }),
  setActiveSession: (session) => set({ activeSession: session, messages: [] }),
  addSession: (session) =>
    set((state) => ({ sessions: [session, ...state.sessions] })),
  updateSession: (session) =>
    set((state) => ({
      sessions: state.sessions.map((s) => (s.id === session.id ? session : s)),
      activeSession: state.activeSession?.id === session.id ? session : state.activeSession,
    })),
  removeSession: (id) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== id),
      activeSession: state.activeSession?.id === id ? null : state.activeSession,
    })),
  setMessages: (messages) => set({ messages }),
  appendMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  // 按 id 更新，避免并发/重复提交时把 token 写进别的消息气泡
  updateMessage: (id, patch) =>
    set((state) => ({
      messages: state.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    })),
}))
