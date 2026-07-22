import { create } from 'zustand'
import type { KnowledgeBase } from '../types/api'

interface KBStore {
  kbs: KnowledgeBase[]
  activeKB: KnowledgeBase | null
  setKBs: (kbs: KnowledgeBase[]) => void
  setActiveKB: (kb: KnowledgeBase | null) => void
  addKB: (kb: KnowledgeBase) => void
  removeKB: (id: string) => void
}

export const useKBStore = create<KBStore>((set) => ({
  kbs: [],
  activeKB: null,
  setKBs: (kbs) => set({ kbs }),
  setActiveKB: (kb) => set({ activeKB: kb }),
  addKB: (kb) => set((state) => ({ kbs: [...state.kbs, kb] })),
  removeKB: (id) =>
    set((state) => ({
      kbs: state.kbs.filter((kb) => kb.id !== id),
      activeKB: state.activeKB?.id === id ? null : state.activeKB,
    })),
}))
