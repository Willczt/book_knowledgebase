import { create } from 'zustand'
import type { Session, Message } from '../types/api'

interface SessionStore {
  sessions: Session[]
  activeSession: Session | null
  messages: Message[]
  setSessions: (sessions: Session[]) => void
  setActiveSession: (session: Session | null) => void
  addSession: (session: Session) => void
  removeSession: (id: string) => void
  setMessages: (messages: Message[]) => void
  appendMessage: (message: Message) => void
  updateLastAssistantMessage: (content: string) => void
}

export const useSessionStore = create<SessionStore>((set) => ({
  sessions: [],
  activeSession: null,
  messages: [],
  setSessions: (sessions) => set({ sessions }),
  setActiveSession: (session) => set({ activeSession: session, messages: [] }),
  addSession: (session) =>
    set((state) => ({ sessions: [session, ...state.sessions] })),
  removeSession: (id) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== id),
      activeSession: state.activeSession?.id === id ? null : state.activeSession,
    })),
  setMessages: (messages) => set({ messages }),
  appendMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  updateLastAssistantMessage: (content) =>
    set((state) => {
      const msgs = [...state.messages]
      const last = msgs[msgs.length - 1]
      if (last && last.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, content }
      }
      return { messages: msgs }
    }),
}))
