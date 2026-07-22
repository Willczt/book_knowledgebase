import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2, Upload, X, ChevronDown, ChevronRight, RefreshCw, Database } from 'lucide-react'
import { knowledgeBaseApi } from '../../api/knowledgeBase'
import { documentsApi } from '../../api/documents'
import { sessionsApi } from '../../api/sessions'
import { useKBStore } from '../../stores/kbStore'
import { useSessionStore } from '../../stores/sessionStore'
import { ChatWindow } from '../chat/ChatWindow'
import type { EmbeddingModelInfo, LLMProviderInfo } from '../../types/api'
import clsx from 'clsx'

function CreateKBModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [provider, setProvider] = useState('doubao')
  const [modelId, setModelId] = useState('doubao-embedding-text-240715')
  const [dim, setDim] = useState(2048)
  const queryClient = useQueryClient()
  const addKB = useKBStore((s) => s.addKB)

  const { data: models } = useQuery<EmbeddingModelInfo[]>({
    queryKey: ['embedding-models'],
    queryFn: async () => {
      const r = await fetch('/api/v1/embedding-models')
      return r.json()
    },
  })

  const mutation = useMutation({
    mutationFn: () =>
      knowledgeBaseApi.create({
        name,
        description: desc || undefined,
        embedding_provider: provider,
        embedding_model_id: modelId,
        embedding_vector_dim: dim,
      }),
    onSuccess: (kb) => {
      addKB(kb)
      queryClient.invalidateQueries({ queryKey: ['kbs'] })
      onClose()
    },
  })

  const handleSelectModel = (m: EmbeddingModelInfo) => {
    setProvider(m.provider)
    setModelId(m.model_id)
    setDim(m.vector_dim)
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="card w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-semibold">新建知识库</h2>
          <button onClick={onClose} className="btn-ghost p-1"><X size={16} /></button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-xs text-gray-400 mb-1 block">名称 *</label>
            <input
              className="input-base w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="如：机器学习书单"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-1 block">描述</label>
            <textarea
              className="input-base w-full resize-none"
              rows={2}
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="可选"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-2 block">
              Embedding 模型（创建后不可更改）
            </label>
            <div className="space-y-2">
              {models?.map((m) => (
                <label
                  key={`${m.provider}/${m.model_id}`}
                  className={clsx(
                    'flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors',
                    provider === m.provider && modelId === m.model_id
                      ? 'border-brand bg-brand/10'
                      : 'border-gray-700 hover:border-gray-600',
                  )}
                >
                  <input
                    type="radio"
                    className="mt-0.5"
                    checked={provider === m.provider && modelId === m.model_id}
                    onChange={() => handleSelectModel(m)}
                  />
                  <div className="text-xs">
                    <div className="font-medium text-gray-200">{m.model_id}</div>
                    <div className="text-gray-500">{m.description}</div>
                    <div className="text-gray-600 mt-0.5">dim={m.vector_dim}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className="btn-ghost text-sm">取消</button>
          <button
            onClick={() => mutation.mutate()}
            disabled={!name.trim() || mutation.isPending}
            className="btn-primary text-sm"
          >
            {mutation.isPending ? '创建中...' : '创建'}
          </button>
        </div>
        {mutation.isError && (
          <p className="text-red-400 text-xs mt-2">{mutation.error.message}</p>
        )}
      </div>
    </div>
  )
}

function DocumentUploader({ kbId }: { kbId: string }) {
  const [uploading, setUploading] = useState(false)
  const queryClient = useQueryClient()

  const handleFiles = async (files: FileList) => {
    setUploading(true)
    for (const file of Array.from(files)) {
      try {
        const doc = await documentsApi.upload(kbId, file)
        // 轮询状态
        const poll = async () => {
          const status = await documentsApi.status(kbId, doc.id)
          if (status.status === 'indexed' || status.status === 'failed') {
            queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
          } else {
            setTimeout(poll, 2000)
          }
        }
        poll()
      } catch (e) {
        console.error(e)
      }
    }
    setUploading(false)
    queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
  }

  return (
    <label className="flex flex-col items-center justify-center border-2 border-dashed border-gray-700 
                       rounded-xl py-6 px-4 cursor-pointer hover:border-brand/50 transition-colors">
      <Upload size={20} className="text-gray-500 mb-2" />
      <span className="text-xs text-gray-500">
        {uploading ? '上传中...' : '点击或拖拽上传 PDF / EPUB / Markdown'}
      </span>
      <input
        type="file"
        accept=".pdf,.epub,.md,.markdown"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && handleFiles(e.target.files)}
        disabled={uploading}
      />
    </label>
  )
}

export function Sidebar() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [expandedKB, setExpandedKB] = useState<string | null>(null)
  const { kbs, setKBs, activeKB, setActiveKB, removeKB } = useKBStore()
  const { sessions, setSessions, activeSession, setActiveSession, addSession, removeSession } =
    useSessionStore()
  const queryClient = useQueryClient()

  const { data: llmProviders } = useQuery<LLMProviderInfo[]>({
    queryKey: ['llm-providers'],
    queryFn: async () => {
      const r = await fetch('/api/v1/llm-providers')
      return r.json()
    },
  })

  useQuery({
    queryKey: ['kbs'],
    queryFn: async () => {
      const data = await knowledgeBaseApi.list()
      setKBs(data)
      return data
    },
  })

  const { data: docs } = useQuery({
    queryKey: ['documents', expandedKB],
    queryFn: () => (expandedKB ? documentsApi.list(expandedKB) : Promise.resolve([])),
    enabled: !!expandedKB,
  })

  const loadSessions = async (kbId: string) => {
    const data = await sessionsApi.list(kbId)
    setSessions(data)
  }

  const handleSelectKB = async (kbId: string) => {
    const kb = kbs.find((k) => k.id === kbId)
    if (!kb) return
    setActiveKB(kb)
    setActiveSession(null)
    await loadSessions(kbId)
  }

  const handleNewSession = async () => {
    if (!activeKB || !llmProviders?.[0]) return
    const lp = llmProviders[0]
    const session = await sessionsApi.create({
      kb_id: activeKB.id,
      llm_provider: lp.provider,
      llm_model_id: lp.model_id,
    })
    addSession(session)
    setActiveSession(session)
  }

  const handleDeleteKB = async (kbId: string) => {
    if (!confirm('确定删除此知识库？（含所有文档和会话）')) return
    await knowledgeBaseApi.delete(kbId)
    removeKB(kbId)
  }

  const handleClearKB = async (kbId: string) => {
    if (!confirm('确定清空向量库？')) return
    await knowledgeBaseApi.clear(kbId)
    queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
  }

  return (
    <aside className="w-64 h-full flex flex-col bg-gray-900 border-r border-gray-800">
      {/* Header */}
      <div className="px-4 py-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-3">
          <Database size={16} className="text-brand" />
          <span className="font-semibold text-sm">书库知识库</span>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary w-full text-xs flex items-center justify-center gap-1.5"
        >
          <Plus size={14} />
          新建知识库
        </button>
      </div>

      {/* KB List */}
      <div className="flex-1 overflow-y-auto py-2">
        {kbs.map((kb) => (
          <div key={kb.id}>
            {/* KB 行：chevron 独立控制展开，名字点击选中 */}
            <div
              className={clsx(
                'flex items-center gap-1 px-2 py-2 rounded-lg mx-1 group',
                activeKB?.id === kb.id ? 'bg-gray-800' : 'hover:bg-gray-800/50',
              )}
            >
              {/* chevron：仅控制文档面板展开/收起 */}
              <button
                onClick={() => setExpandedKB(expandedKB === kb.id ? null : kb.id)}
                className="p-0.5 text-gray-500 hover:text-gray-300 shrink-0"
              >
                {expandedKB === kb.id ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              </button>
              {/* KB 名字：点击选中（切换活跃 KB，加载会话） */}
              <button
                onClick={() => handleSelectKB(kb.id)}
                className="flex-1 text-left text-xs font-medium text-gray-200 truncate"
              >
                {kb.name}
              </button>
              {/* 操作按钮（hover 显示） */}
              <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 shrink-0">
                <button
                  onClick={() => handleClearKB(kb.id)}
                  className="p-0.5 text-gray-500 hover:text-yellow-400 transition-colors"
                  title="清空向量库"
                >
                  <RefreshCw size={11} />
                </button>
                <button
                  onClick={() => handleDeleteKB(kb.id)}
                  className="p-0.5 text-gray-500 hover:text-red-400 transition-colors"
                  title="删除知识库"
                >
                  <Trash2 size={11} />
                </button>
              </div>
            </div>

            {/* 文档面板（仅展开时显示） */}
            {expandedKB === kb.id && (
              <div className="mx-2 mb-2 space-y-1.5">
                <DocumentUploader kbId={kb.id} />
                {docs?.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center gap-1.5 px-2 py-1 bg-gray-800/50 rounded-lg text-xs"
                  >
                    <span
                      className={clsx(
                        'w-1.5 h-1.5 rounded-full shrink-0',
                        doc.status === 'indexed' ? 'bg-green-500' :
                        doc.status === 'processing' ? 'bg-yellow-500 animate-pulse' :
                        doc.status === 'failed' ? 'bg-red-500' : 'bg-gray-500',
                      )}
                    />
                    <span className="truncate text-gray-400">{doc.filename}</span>
                    {doc.status === 'indexed' && (
                      <span className="text-gray-600 shrink-0">{doc.chunk_count}块</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* 会话列表：只要该 KB 是活跃 KB 就显示，与文档面板无关 */}
            {activeKB?.id === kb.id && (
              <div className="mx-2 mb-2">
                <button
                  onClick={handleNewSession}
                  className="w-full text-xs text-gray-500 hover:text-gray-200 flex items-center gap-1.5 py-1.5 px-2 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <Plus size={12} />
                  新对话
                </button>
                {sessions.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => setActiveSession(s)}
                    className={clsx(
                      'flex items-center gap-1.5 px-2 py-1.5 rounded-lg cursor-pointer group text-xs',
                      activeSession?.id === s.id
                        ? 'bg-brand/20 text-brand-light'
                        : 'text-gray-500 hover:bg-gray-800 hover:text-gray-300',
                    )}
                  >
                    <span className="flex-1 truncate">{s.title || '新会话'}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        sessionsApi.delete(s.id).then(() => removeSession(s.id))
                      }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-red-400"
                    >
                      <X size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {showCreateModal && <CreateKBModal onClose={() => setShowCreateModal(false)} />}
    </aside>
  )
}

export function MainContent() {
  const { activeSession, activeKB } = (() => ({
    activeSession: useSessionStore((s) => s.activeSession),
    activeKB: useKBStore((s) => s.activeKB),
  }))()

  if (!activeKB) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600">
        <div className="text-center">
          <div className="text-5xl mb-4">📚</div>
          <p className="text-gray-400 font-medium">选择或创建一个知识库</p>
          <p className="text-sm mt-2">然后上传 PDF、EPUB 或 Markdown 文档开始问答</p>
        </div>
      </div>
    )
  }

  if (!activeSession) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600">
        <div className="text-center">
          <div className="text-5xl mb-4">💬</div>
          <p className="text-gray-400 font-medium">选择或新建一个会话</p>
          <p className="text-sm mt-1">点击左侧「新对话」开始</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* TopBar */}
      <header className="flex items-center px-5 py-3 border-b border-gray-800 bg-gray-900">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-200 truncate">
            {activeSession.title || '新会话'}
          </p>
          <p className="text-xs text-gray-500 truncate">
            {activeKB.name} · {activeSession.llm_provider}/{activeSession.llm_model_id}
          </p>
        </div>
      </header>
      <ChatWindow sessionId={activeSession.id} />
    </div>
  )
}
