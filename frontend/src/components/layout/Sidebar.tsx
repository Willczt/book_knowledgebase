import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus, Trash2, Upload, Download, Pencil, X, ChevronDown, ChevronRight,
  RefreshCw, Database, BookOpen, MessageSquare, Sparkles,
} from 'lucide-react'
import { knowledgeBaseApi } from '../../api/knowledgeBase'
import { documentsApi } from '../../api/documents'
import { sessionsApi } from '../../api/sessions'
import { systemApi } from '../../api/system'
import { useKBStore } from '../../stores/kbStore'
import { useSessionStore } from '../../stores/sessionStore'
import { ChatWindow } from '../chat/ChatWindow'
import { APP_NAME, APP_TAGLINE } from '../../branding'
import type {
  Document,
  EmbeddingModelInfo,
  HealthResponse,
  IngestProgress,
  KnowledgeBase,
  LLMProviderInfo,
  Session,
} from '../../types/api'
import clsx from 'clsx'

/** 触发浏览器下载 */
function downloadBlob(data: Blob, filename: string) {
  const url = URL.createObjectURL(data)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** 行内重命名输入框：Enter 提交，Esc 取消，失焦提交 */
function InlineRename({
  value,
  onChange,
  onSubmit,
  onCancel,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onCancel: () => void
}) {
  // Enter 提交后输入框会卸载并触发一次 blur，用 ref 防止重复提交
  const submitted = useRef(false)
  const submit = () => {
    if (submitted.current) return
    submitted.current = true
    onSubmit()
  }

  return (
    <input
      className="flex-1 min-w-0 bg-white/[0.06] border border-brand/40 rounded-md px-1.5 py-0.5 text-xs text-gray-100 outline-none focus:border-brand"
      value={value}
      autoFocus
      onChange={(e) => onChange(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onBlur={submit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') submit()
        if (e.key === 'Escape') {
          submitted.current = true
          onCancel()
        }
      }}
    />
  )
}

function CreateKBModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [provider, setProvider] = useState('doubao')
  const [modelId, setModelId] = useState('doubao-embedding-vision-251215')
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

  // 默认选中第一个「当前真正可用」的模型（例如去掉 Ark Key 后不再默认豆包）
  useEffect(() => {
    if (!models?.length) return
    const exists = models.some((m) => m.provider === provider && m.model_id === modelId)
    if (!exists) {
      setProvider(models[0].provider)
      setModelId(models[0].model_id)
      setDim(models[0].vector_dim)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

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
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
      <div className="glass-card w-full max-w-md p-6 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #7c3aed, #3b82f6)' }}
            >
              <Database size={13} className="text-white" />
            </div>
            <h2 className="text-sm font-semibold text-gray-100">新建知识库</h2>
          </div>
          <button onClick={onClose} className="btn-icon"><X size={15} /></button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block font-medium">名称 *</label>
            <input
              className="input-base w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="如：机器学习书单"
              autoFocus
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block font-medium">描述</label>
            <textarea
              className="input-base w-full resize-none"
              rows={2}
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="可选"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-2 block font-medium">
              Embedding 模型
              <span className="ml-1.5 text-[10px] text-amber-500/80 font-normal">创建后不可更改</span>
            </label>
            <div className="space-y-2">
              {models?.map((m) => (
                <label
                  key={`${m.provider}/${m.model_id}`}
                  className={clsx(
                    'flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-150',
                    provider === m.provider && modelId === m.model_id
                      ? 'border-brand/60 bg-brand/10'
                      : 'border-white/[0.06] hover:border-white/[0.12] bg-surface-2/60',
                  )}
                >
                  <input
                    type="radio"
                    className="mt-0.5 accent-brand"
                    checked={provider === m.provider && modelId === m.model_id}
                    onChange={() => handleSelectModel(m)}
                  />
                  <div className="text-xs">
                    <div className="font-semibold text-gray-200">{m.model_id}</div>
                    <div className="text-gray-500 mt-0.5">{m.description}</div>
                    <div className="text-gray-700 mt-0.5 font-mono">dim={m.vector_dim}</div>
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
            className="btn-primary text-sm px-5"
          >
            {mutation.isPending ? '创建中...' : '创建知识库'}
          </button>
        </div>
        {mutation.isError && (
          <p className="text-red-400 text-xs mt-2 px-1">{mutation.error.message}</p>
        )}
      </div>
    </div>
  )
}

function DocumentUploader({ kbId }: { kbId: string }) {
  const [uploading, setUploading] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const queryClient = useQueryClient()

  const handleFiles = async (files: FileList) => {
    setUploading(true)
    for (const file of Array.from(files)) {
      try {
        const doc = await documentsApi.upload(kbId, file)
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
    <label
      className={clsx(
        'flex flex-col items-center justify-center rounded-xl py-4 px-3 cursor-pointer transition-all duration-200 border-2 border-dashed',
        isDragOver
          ? 'border-brand/60 bg-brand/10'
          : 'border-white/[0.08] hover:border-brand/40 hover:bg-brand/5',
      )}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setIsDragOver(false); e.dataTransfer.files && handleFiles(e.dataTransfer.files) }}
    >
      <Upload size={16} className={clsx('mb-1.5 transition-colors', isDragOver ? 'text-brand-light' : 'text-gray-600')} />
      <span className="text-[11px] text-gray-600 text-center">
        {uploading ? (
          <span className="text-brand-light animate-pulse">上传中...</span>
        ) : (
          <>点击或拖拽文件<br /><span className="text-gray-700">PDF · EPUB · Markdown · TXT</span></>
        )}
      </span>
      <input
        type="file"
        accept=".pdf,.epub,.md,.markdown,.txt,.text,text/plain"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && handleFiles(e.target.files)}
        disabled={uploading}
      />
    </label>
  )
}

/**
 * 轮询处理中文档的摄入进度。
 * 只对 pending/processing 的文档发请求，全部处理完就自动停下。
 */
function useIngestProgress(
  kbId: string | null,
  docs: Document[] | undefined,
  onFinished: () => void,
): Record<string, IngestProgress> {
  const [progressMap, setProgressMap] = useState<Record<string, IngestProgress>>({})
  const finishedRef = useRef(false)

  const pendingIds = (docs ?? [])
    .filter((d) => d.status === 'pending' || d.status === 'processing')
    .map((d) => d.id)
  const pendingKey = pendingIds.join(',')

  useEffect(() => {
    if (!kbId || !pendingKey) {
      setProgressMap({})
      return
    }
    const ids = pendingKey.split(',')
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    finishedRef.current = false

    const tick = async () => {
      const next: Record<string, IngestProgress> = {}
      let allDone = true
      await Promise.all(
        ids.map(async (id) => {
          try {
            const status = await documentsApi.status(kbId, id)
            if (status.progress) next[id] = status.progress
            if (status.status !== 'indexed' && status.status !== 'failed') allDone = false
          } catch {
            allDone = false
          }
        }),
      )
      if (cancelled) return
      setProgressMap(next)
      if (allDone) {
        if (!finishedRef.current) {
          finishedRef.current = true
          onFinished()
        }
        return
      }
      timer = setTimeout(tick, 1500)
    }

    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbId, pendingKey])

  return progressMap
}

export function Sidebar() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [expandedKB, setExpandedKB] = useState<string | null>(null)
  const [editingKB, setEditingKB] = useState<{ id: string; value: string } | null>(null)
  const [editingSession, setEditingSession] = useState<{ id: string; value: string } | null>(null)
  const [busyIds, setBusyIds] = useState<Record<string, boolean>>({})
  const kbImportRef = useRef<HTMLInputElement>(null)
  const sessionImportRef = useRef<HTMLInputElement>(null)
  const { kbs, setKBs, activeKB, setActiveKB, removeKB, updateKB } = useKBStore()
  const {
    sessions, setSessions, activeSession, setActiveSession,
    addSession, removeSession, updateSession,
  } = useSessionStore()
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

  // 摄入进度：对处理中的文档轮询 status，拿到 percent / 当前分片数
  const progressMap = useIngestProgress(expandedKB, docs, () => {
    queryClient.invalidateQueries({ queryKey: ['documents', expandedKB] })
  })

  // 外部模型服务连通性（30s 刷新一次，不阻塞界面）
  const { data: health } = useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: systemApi.health,
    refetchInterval: 30_000,
    retry: false,
  })

  const loadSessions = async (kbId: string) => {
    const data = await sessionsApi.list(kbId)
    setSessions(data)
  }

  // 移除摄入失败的文档（含服务重启后残留的僵尸任务）
  const handleDeleteDoc = async (kbId: string, docId: string) => {
    try {
      await documentsApi.delete(kbId, docId)
    } catch (e) {
      console.error(e)
    }
    queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
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

  const markBusy = (key: string, value: boolean) =>
    setBusyIds((prev) => ({ ...prev, [key]: value }))

  const handleRenameKB = async (kb: KnowledgeBase, value: string) => {
    setEditingKB(null)
    const name = value.trim()
    if (!name || name === kb.name) return
    try {
      const updated = await knowledgeBaseApi.update(kb.id, { name })
      updateKB(updated)
      queryClient.invalidateQueries({ queryKey: ['kbs'] })
    } catch (e) {
      alert(`重命名失败：${(e as Error).message}`)
    }
  }

  const handleRenameSession = async (session: Session, value: string) => {
    setEditingSession(null)
    const title = value.trim()
    if (!title || title === session.title) return
    try {
      const updated = await sessionsApi.update(session.id, { title })
      updateSession(updated)
    } catch (e) {
      alert(`重命名失败：${(e as Error).message}`)
    }
  }

  const handleExportKB = async (kb: KnowledgeBase) => {
    if (busyIds[kb.id]) return
    markBusy(kb.id, true)
    try {
      const blob = await knowledgeBaseApi.exportArchive(kb.id)
      downloadBlob(blob, `${kb.name}.kbz.zip`)
    } catch (e) {
      alert(`导出失败：${(e as Error).message}`)
    } finally {
      markBusy(kb.id, false)
    }
  }

  const handleImportKB = async (file: File) => {
    try {
      const result = await knowledgeBaseApi.importArchive(file)
      const list = await knowledgeBaseApi.list()
      setKBs(list)
      queryClient.invalidateQueries({ queryKey: ['kbs'] })
      const imported = list.find((k) => k.id === result.id)
      if (imported) {
        setActiveKB(imported)
        setExpandedKB(imported.id)
        await loadSessions(imported.id)
      }
      alert(
        `导入完成：${result.documents} 个文档 · ${result.vectors} 条向量 · ` +
        `${result.sessions} 个会话 · ${result.messages} 条消息`,
      )
    } catch (e) {
      // 409：知识库已存在（同一归档不能重复导入）
      alert((e as Error).message)
    }
  }

  const handleExportSession = async (session: Session) => {
    try {
      const blob = await sessionsApi.exportArchive(session.id)
      downloadBlob(blob, `${session.title || 'session'}.session.zip`)
    } catch (e) {
      alert(`导出失败：${(e as Error).message}`)
    }
  }

  const handleImportSession = async (kbId: string, file: File) => {
    try {
      await sessionsApi.importArchive(kbId, file)
      await loadSessions(kbId)
    } catch (e) {
      alert(`会话导入失败：${(e as Error).message}`)
    }
  }

  return (
    <aside
      className="w-[268px] h-full flex flex-col shrink-0"
      style={{
        background: 'linear-gradient(180deg, #080a12 0%, #07090f 100%)',
        borderRight: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      {/* Logo Header */}
      <div className="px-4 py-4 shrink-0" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="flex items-center gap-2.5 mb-4">
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #3b82f6)', boxShadow: '0 4px 14px rgba(124,58,237,0.4)' }}
          >
            <BookOpen size={15} className="text-white" />
          </div>
          <div>
            <div className="text-[13px] font-bold gradient-text leading-tight">{APP_NAME}</div>
            <div className="text-[10px] text-gray-700 leading-tight">{APP_TAGLINE}</div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary flex-1 text-xs flex items-center justify-center gap-1.5 py-2"
          >
            <Plus size={13} />
            新建知识库
          </button>
          <button
            onClick={() => kbImportRef.current?.click()}
            className="btn-ghost p-2 shrink-0"
            title="导入知识库（.kbz.zip）"
          >
            <Upload size={13} />
          </button>
          <input
            ref={kbImportRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              e.target.value = ''
              if (file) handleImportKB(file)
            }}
          />
        </div>
      </div>

      {/* KB List */}
      <div className="flex-1 overflow-y-auto py-2 px-2">
        {kbs.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10 gap-2">
            <Database size={22} className="text-gray-700" />
            <p className="text-xs text-gray-700 text-center">暂无知识库<br />点击上方按钮创建</p>
          </div>
        )}

        {kbs.map((kb) => (
          <div key={kb.id} className="mb-1">
            {/* KB Row */}
            <div
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-2 rounded-xl group transition-all duration-150',
                activeKB?.id === kb.id
                  ? 'bg-brand/10 border border-brand/20'
                  : 'hover:bg-white/[0.04] border border-transparent',
              )}
            >
              {/* Expand chevron */}
              <button
                onClick={() => setExpandedKB(expandedKB === kb.id ? null : kb.id)}
                className="p-0.5 text-gray-600 hover:text-gray-400 shrink-0 transition-colors"
              >
                {expandedKB === kb.id
                  ? <ChevronDown size={11} />
                  : <ChevronRight size={11} />}
              </button>

              {/* KB icon */}
              <div className={clsx(
                'w-5 h-5 rounded-md flex items-center justify-center shrink-0',
                activeKB?.id === kb.id ? 'bg-brand/20' : 'bg-white/[0.05]',
              )}>
                <Database size={10} className={activeKB?.id === kb.id ? 'text-brand-light' : 'text-gray-500'} />
              </div>

              {/* Name */}
              {editingKB?.id === kb.id ? (
                <InlineRename
                  value={editingKB.value}
                  onChange={(value) => setEditingKB({ id: kb.id, value })}
                  onSubmit={() => handleRenameKB(kb, editingKB.value)}
                  onCancel={() => setEditingKB(null)}
                />
              ) : (
                <button
                  onClick={() => handleSelectKB(kb.id)}
                  onDoubleClick={() => setEditingKB({ id: kb.id, value: kb.name })}
                  title={kb.name}
                  className={clsx(
                    'flex-1 min-w-0 text-left text-xs font-medium truncate transition-colors',
                    activeKB?.id === kb.id ? 'text-gray-100' : 'text-gray-400 hover:text-gray-200',
                  )}
                >
                  {kb.name}
                </button>
              )}

              {/* Actions */}
              <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 shrink-0 transition-opacity">
                <button
                  onClick={() => setEditingKB({ id: kb.id, value: kb.name })}
                  className="btn-icon p-1"
                  title="重命名"
                >
                  <Pencil size={10} />
                </button>
                <button
                  onClick={() => handleExportKB(kb)}
                  className={clsx('btn-icon p-1', busyIds[kb.id] && 'animate-pulse')}
                  title="导出知识库（含向量库）"
                  disabled={!!busyIds[kb.id]}
                >
                  <Download size={10} />
                </button>
                <button
                  onClick={() => handleClearKB(kb.id)}
                  className="btn-icon p-1"
                  title="清空向量库"
                >
                  <RefreshCw size={10} />
                </button>
                <button
                  onClick={() => handleDeleteKB(kb.id)}
                  className="p-1 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-400/10 transition-all"
                  title="删除知识库"
                >
                  <Trash2 size={10} />
                </button>
              </div>
            </div>

            {/* Document Panel */}
            {expandedKB === kb.id && (
              <div className="mx-1 mb-2 mt-1 space-y-1.5 animate-fade-in">
                <DocumentUploader kbId={kb.id} />
                {docs?.map((doc) => {
                  const progress = progressMap[doc.id]
                  const percent = Math.max(0, Math.min(100, progress?.percent ?? 0))
                  const detail = progress
                    ? progress.current != null && progress.total
                      ? `${progress.stage_label} ${progress.current}/${progress.total}`
                      : progress.stage_label
                    : '等待处理…'
                  return (
                  <div
                    key={doc.id}
                    className="px-2.5 py-1.5 bg-surface-2/60 rounded-lg text-xs border border-white/[0.04]"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={clsx(
                          'w-1.5 h-1.5 rounded-full shrink-0',
                          doc.status === 'indexed' ? 'bg-emerald-500' :
                          doc.status === 'processing' ? 'bg-amber-500 animate-pulse-slow' :
                          doc.status === 'failed' ? 'bg-red-500' : 'bg-gray-600',
                        )}
                      />
                      <span
                        className="truncate text-gray-500 flex-1"
                        title={doc.status === 'failed' ? doc.error_message || '摄入失败' : doc.filename}
                      >
                        {doc.filename}
                      </span>
                      {doc.status === 'indexed' && (
                        <span className="text-gray-700 shrink-0 text-[10px]">{doc.chunk_count}</span>
                      )}
                      {doc.status === 'failed' && (
                        <button
                          onClick={() => handleDeleteDoc(kb.id, doc.id)}
                          title={doc.error_message || '摄入失败，点击移除'}
                          className="shrink-0 text-gray-600 hover:text-red-400 transition-colors"
                        >
                          <X size={11} />
                        </button>
                      )}
                    </div>

                    {(doc.status === 'pending' || doc.status === 'processing') && (
                      <div className="mt-1.5">
                        <div className="flex items-center justify-between gap-2 text-[10px]">
                          <span className="truncate text-gray-600">{detail}</span>
                          <span className="shrink-0 text-brand-light/80 font-mono">
                            {Math.round(percent)}%
                          </span>
                        </div>
                        <div className="mt-1 h-1 rounded-full bg-white/[0.06] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-brand to-blue-500 transition-[width] duration-500 ease-out"
                            style={{ width: `${percent}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  )
                })}
              </div>
            )}

            {/* Session List */}
            {activeKB?.id === kb.id && (
              <div className="mx-1 mb-2 animate-fade-in">
                {/* Divider */}
                <div className="flex items-center gap-2 px-1 mb-1.5">
                  <div className="h-px flex-1 bg-white/[0.04]" />
                  <span className="text-[10px] text-gray-700 font-medium">对话</span>
                  <div className="h-px flex-1 bg-white/[0.04]" />
                  <button
                    onClick={() => sessionImportRef.current?.click()}
                    className="p-0.5 text-gray-700 hover:text-brand-light transition-colors"
                    title="导入会话（.session.zip）"
                  >
                    <Upload size={9} />
                  </button>
                  <input
                    ref={sessionImportRef}
                    type="file"
                    accept=".zip"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      e.target.value = ''
                      if (file && kb.id) handleImportSession(kb.id, file)
                    }}
                  />
                </div>

                <button
                  onClick={handleNewSession}
                  className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-xl text-xs text-gray-500 hover:text-gray-200 hover:bg-white/[0.05] transition-all duration-150 group"
                >
                  <div className="w-4 h-4 rounded-md bg-white/[0.06] flex items-center justify-center group-hover:bg-brand/20 transition-colors">
                    <Plus size={9} className="group-hover:text-brand-light transition-colors" />
                  </div>
                  新对话
                </button>

                {sessions.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => setActiveSession(s)}
                    className={clsx(
                      'flex items-center gap-2 px-2.5 py-1.5 rounded-xl cursor-pointer group text-xs transition-all duration-150',
                      activeSession?.id === s.id
                        ? 'bg-brand/15 text-brand-light border border-brand/20'
                        : 'text-gray-500 hover:bg-white/[0.04] hover:text-gray-300 border border-transparent',
                    )}
                  >
                    <MessageSquare size={10} className="shrink-0 opacity-60" />
                    {editingSession?.id === s.id ? (
                      <InlineRename
                        value={editingSession.value}
                        onChange={(value) => setEditingSession({ id: s.id, value })}
                        onSubmit={() => handleRenameSession(s, editingSession.value)}
                        onCancel={() => setEditingSession(null)}
                      />
                    ) : (
                      <span
                        className="flex-1 truncate"
                        onDoubleClick={(e) => {
                          e.stopPropagation()
                          setEditingSession({ id: s.id, value: s.title || '' })
                        }}
                        title={s.title || '新会话'}
                      >
                        {s.title || '新会话'}
                      </span>
                    )}
                    <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 shrink-0 transition-opacity">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setEditingSession({ id: s.id, value: s.title || '' })
                        }}
                        className="p-0.5 hover:text-brand-light transition-all"
                        title="重命名会话"
                      >
                        <Pencil size={9} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleExportSession(s)
                        }}
                        className="p-0.5 hover:text-brand-light transition-all"
                        title="导出会话"
                      >
                        <Download size={9} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          sessionsApi.delete(s.id).then(() => removeSession(s.id))
                        }}
                        className="p-0.5 hover:text-red-400 transition-all"
                        title="删除会话"
                      >
                        <X size={9} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div
        className="px-4 py-3 shrink-0"
        style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
      >
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 mb-1.5">
          {([
            ['Ollama', health?.providers?.ollama],
            ['DeepSeek', health?.providers?.deepseek],
            ['豆包', health?.providers?.ark],
          ] as const).map(([label, provider]) => (
            <span
              key={label}
              className="flex items-center gap-1 text-[10px] text-gray-600"
              title={
                provider
                  ? provider.ok
                    ? `${label} 已连通${provider.latency_ms != null ? ` · ${provider.latency_ms}ms` : ''}`
                    : `${label} 不可用：${provider.error || (provider.configured === false ? '未配置' : '状态未知')}`
                  : `${label} 状态未知`
              }
            >
              <span
                className={clsx(
                  'w-1.5 h-1.5 rounded-full',
                  !provider ? 'bg-gray-700' : provider.ok ? 'bg-emerald-500/80' : 'bg-red-500/80',
                )}
              />
              {label}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Sparkles size={11} className="text-brand/60" />
          <span className="text-[10px] text-gray-700">Hybrid Search · BM25 + Dense · RRF</span>
        </div>
      </div>

      {showCreateModal && <CreateKBModal onClose={() => setShowCreateModal(false)} />}
    </aside>
  )
}

export function MainContent() {
  const activeSession = useSessionStore((s) => s.activeSession)
  const activeKB = useKBStore((s) => s.activeKB)

  if (!activeKB) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface-0">
        <div className="text-center animate-fade-in">
          {/* Gradient orbs */}
          <div className="relative mb-6 flex justify-center">
            <div className="absolute w-32 h-32 rounded-full blur-3xl opacity-20"
              style={{ background: 'radial-gradient(circle, #7c3aed, transparent)' }} />
            <div className="absolute w-32 h-32 rounded-full blur-3xl opacity-20 translate-x-8"
              style={{ background: 'radial-gradient(circle, #3b82f6, transparent)' }} />
            <div
              className="relative w-16 h-16 rounded-2xl flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(59,130,246,0.2))', border: '1px solid rgba(124,58,237,0.3)' }}
            >
              <BookOpen size={28} className="text-brand-light" />
            </div>
          </div>
          <h2 className="text-lg font-semibold gradient-text mb-2">{APP_NAME}</h2>
          <p className="text-[11px] tracking-[0.3em] text-gray-700 uppercase mb-2">Langhuan</p>
          <p className="text-sm text-gray-600 max-w-xs leading-relaxed">
            从左侧选择知识库，或创建新的知识库并上传 PDF、EPUB、Markdown、TXT 文档
          </p>
        </div>
      </div>
    )
  }

  if (!activeSession) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface-0">
        <div className="text-center animate-fade-in">
          <div
            className="w-12 h-12 rounded-xl mx-auto mb-4 flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(59,130,246,0.2))', border: '1px solid rgba(124,58,237,0.25)' }}
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
    <div className="flex-1 flex flex-col overflow-hidden bg-surface-0">
      {/* Header */}
      <header
        className="flex items-center px-6 py-3 shrink-0"
        style={{
          background: 'rgba(7,8,14,0.8)',
          backdropFilter: 'blur(12px)',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
        }}
      >
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
