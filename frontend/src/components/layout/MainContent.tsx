import { BookOpen, MessageSquare } from 'lucide-react'

import { useKBStore } from '../../stores/kbStore'
import { useSessionStore } from '../../stores/sessionStore'
import { APP_LATIN, APP_NAME } from '../../branding'
import { ChatWindow } from '../chat/ChatWindow'

export function MainContent() {
  const activeSession = useSessionStore((s) => s.activeSession)
  const activeKB = useKBStore((s) => s.activeKB)

  if (!activeKB) {
    return (
      <div className="relative z-10 flex-1 flex items-center justify-center">
        <div className="text-center animate-fade-in">
          {/* Gradient orbs */}
          <div className="relative mb-8 flex justify-center">
            <div className="absolute w-36 h-36 rounded-full blur-3xl opacity-30 -translate-y-2"
              style={{ background: 'radial-gradient(circle, #7c3aed, transparent)' }} />
            <div className="absolute w-36 h-36 rounded-full blur-3xl opacity-25 translate-x-10"
              style={{ background: 'radial-gradient(circle, #3b82f6, transparent)' }} />
            <div
              className="relative w-16 h-16 rounded-2xl flex items-center justify-center"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(167,139,250,0.35)',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.12), 0 0 26px -12px rgba(124,58,237,0.95)',
              }}
            >
              <span className="halo-ring" aria-hidden="true" />
              <BookOpen size={28} className="relative text-brand-light" />
            </div>
          </div>
          <h2 className="text-lg font-semibold gradient-text mb-2">{APP_NAME}</h2>
          <p className="text-[11px] tracking-[0.3em] text-gray-700 uppercase mb-2">{APP_LATIN}</p>
          <p className="text-sm text-gray-600 max-w-xs leading-relaxed">
            从左侧选择知识库，或创建新的知识库并上传 PDF、EPUB、Markdown、TXT 文档
          </p>
        </div>
      </div>
    )
  }

  if (!activeSession) {
    return (
      <div className="relative z-10 flex-1 flex items-center justify-center">
        <div className="text-center animate-fade-in">
          <div
            className="relative w-12 h-12 rounded-xl mx-auto mb-4 flex items-center justify-center"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(167,139,250,0.32)',
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.12), 0 0 24px -12px rgba(124,58,237,0.9)',
            }}
          >
            <MessageSquare size={20} className="text-brand-light" />
          </div>
          <p className="text-sm font-medium text-gray-400">{activeKB.name}</p>
          <p className="text-xs text-gray-600 mt-1">点击左侧「新对话」开始提问</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative z-10 flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="glass-header flex items-center px-6 py-3 shrink-0">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-200 truncate">
            {activeSession.title || '新会话'}
          </p>
          <p className="text-[11px] text-gray-600 truncate mt-0.5">
            <span className="text-brand/70">{activeKB.name}</span>
            <span className="mx-1.5 text-gray-700">·</span>
            {activeSession.llm_provider}/{activeSession.llm_model_id}
          </p>
        </div>
      </header>

      <ChatWindow sessionId={activeSession.id} />
    </div>
  )
}
