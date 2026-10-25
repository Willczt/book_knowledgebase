import { create } from 'zustand'
import type { KnowledgeBase } from '../types/api'

interface KBStore {
  kbs: KnowledgeBase[]
  activeKB: KnowledgeBase | null
  setKBs: (kbs: KnowledgeBase[]) => void
  setActiveKB: (kb: KnowledgeBase | null) => void
  addKB: (kb: KnowledgeBase) => void
  updateKB: (kb: KnowledgeBase) => void
  removeKB: (id: string) => void
}

export const useKBStore = create<KBStore>((set) => ({
  kbs: [],
  activeKB: null,
  setKBs: (kbs) => set({ kbs }),
  setActiveKB: (kb) => set({ activeKB: kb }),
  addKB: (kb) => set((state) => ({ kbs: [...state.kbs, kb] })),
  updateKB: (kb) =>
    set((state) => ({
      kbs: state.kbs.map((item) => (item.id === kb.id ? kb : item)),
      activeKB: state.activeKB?.id === kb.id ? kb : state.activeKB,
    })),
  removeKB: (id) =>
    set((state) => ({
      kbs: state.kbs.filter((kb) => kb.id !== id),
      activeKB: state.activeKB?.id === id ? null : state.activeKB,
    })),
}))
