import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Plus, Trash2, Upload, Download, Pencil, X, ChevronDown, ChevronRight,
  RefreshCw, RotateCcw, Database, BookOpen, MessageSquare, Sparkles, Bug, Loader2,
} from 'lucide-react'
import clsx from 'clsx'

import { knowledgeBaseApi } from '../../api/knowledgeBase'
import { documentsApi } from '../../api/documents'
import { sessionsApi } from '../../api/sessions'
import { systemApi } from '../../api/system'
import { useKBStore } from '../../stores/kbStore'
import { useSessionStore } from '../../stores/sessionStore'
import { APP_NAME, APP_TAGLINE } from '../../branding'
import { downloadBlob } from '../../utils/download'
import { InlineRename } from '../ui/InlineRename'
import { CreateKBModal } from '../modals/CreateKBModal'
import { DocumentUploader, useIngestProgress } from '../documents/DocumentUploader'
import type {
  HealthResponse,
  KnowledgeBase,
  LLMProviderInfo,
  Session,
} from '../../types/api'

interface SidebarProps {
  /** 打开 RAG 全链路调试面板 */
  onOpenDebug?: () => void
}

interface ExportProgress {
  kbId: string
  kbName: string
  jobId: string
  percent: number
  stage: string
  status: 'running' | 'done' | 'failed'
  error?: string
}

export function Sidebar({ onOpenDebug }: SidebarProps) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [expandedKB, setExpandedKB] = useState<string | null>(null)
  const [editingKB, setEditingKB] = useState<{ id: string; value: string } | null>(null)
  const [editingSession, setEditingSession] = useState<{ id: string; value: string } | null>(null)
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null)
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
  const handleReindex = async (kbId: string, docId: string) => {
    try {
      await documentsApi.reindex(kbId, docId)
      queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
    } catch (e) {
      alert((e as Error).message)
    }
  }

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
    try {
      await loadSessions(kbId)
    } catch (e) {
      console.error(e)
      setSessions([])
    }
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
    if (exportProgress?.status === 'running') return
    try {
      const job = await knowledgeBaseApi.startExportJob(kb.id)
      setExportProgress({
        kbId: kb.id,
        kbName: kb.name,
        jobId: job.job_id,
        percent: job.percent,
        stage: job.stage,
        status: job.status,
      })
    } catch (e) {
      alert(`导出失败：${(e as Error).message}`)
    }
  }

  // 导出进度轮询：完成后自动下载，失败则停在卡片上展示原因
  const exportJobId = exportProgress?.jobId
  const exportKbId = exportProgress?.kbId
  const exportStatus = exportProgress?.status
  useEffect(() => {
    if (!exportJobId || !exportKbId || exportStatus !== 'running') return
    let cancelled = false
    let timer: number | undefined

    const tick = async () => {
      try {
        const status = await knowledgeBaseApi.exportJobStatus(exportKbId, exportJobId)
        if (cancelled) return

        if (status.status === 'failed') {
          setExportProgress((prev) =>
            prev
              ? { ...prev, status: 'failed', percent: status.percent, error: status.error || '导出失败' }
              : prev,
          )
          return
        }

        setExportProgress((prev) =>
          prev ? { ...prev, percent: status.percent, stage: status.stage } : prev,
        )

        if (status.status === 'done') {
          const blob = await knowledgeBaseApi.downloadExportJob(exportKbId, exportJobId)
          if (cancelled) return
          downloadBlob(blob, status.filename || `${status.kb_name}.kbz.zip`)
          setExportProgress((prev) =>
            prev ? { ...prev, status: 'done', percent: 100, stage: '下载已开始' } : prev,
          )
          return
        }

        timer = window.setTimeout(tick, 600)
      } catch (e) {
        if (cancelled) return
        setExportProgress((prev) =>
          prev ? { ...prev, status: 'failed', error: (e as Error).message } : prev,
        )
      }
    }

    timer = window.setTimeout(tick, 300)
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [exportJobId, exportKbId, exportStatus])

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
    <aside className="sidebar-panel relative z-10 w-[268px] shrink-0 my-3 ml-3 h-[calc(100%-1.5rem)] flex flex-col">
      {/* Logo Header */}
      <div className="px-4 py-4 shrink-0" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="flex items-center gap-2.5 mb-4">
          <div className="logo-mark w-8 h-8 rounded-xl flex items-center justify-center shrink-0">
            <BookOpen size={15} className="text-brand-light" />
          </div>
          <div>
            <div className="brand-glow text-[13px] font-bold gradient-text leading-tight">{APP_NAME}</div>
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
                'nav-glow sheen flex items-center gap-1.5 px-2.5 py-2 rounded-xl group',
                'transition-all duration-200',
                activeKB?.id === kb.id
                  ? 'is-active bg-brand/10 border border-brand/25'
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
                  className={clsx(
                    'btn-icon p-1',
                    exportProgress?.kbId === kb.id &&
                      exportProgress.status === 'running' &&
                      'animate-pulse',
                  )}
                  title="导出知识库（含向量库）"
                  disabled={exportProgress?.status === 'running'}
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
                      {(doc.status === 'failed' || doc.status === 'pending') && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleReindex(kb.id, doc.id)
                          }}
                          title="重建索引"
                          className="shrink-0 text-gray-600 hover:text-brand-light transition-colors"
                        >
                          <RotateCcw size={11} />
                        </button>
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
                  className="sheen w-full flex items-center gap-2 px-2.5 py-1.5 rounded-xl text-xs text-gray-500 hover:text-gray-200 hover:bg-white/[0.06] transition-all duration-200 group"
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
                      'nav-glow sheen flex items-center gap-2 px-2.5 py-1.5 rounded-xl cursor-pointer group text-xs transition-all duration-200',
                      activeSession?.id === s.id
                        ? 'is-active bg-brand/15 text-brand-light border border-brand/25'
                        : 'text-gray-400 hover:bg-white/[0.05] hover:text-gray-200 border border-transparent',
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
          <span className="text-[10px] text-gray-600 flex-1 truncate">
            Hybrid Search · BM25 + Dense · RRF
          </span>
          <button
            type="button"
            onClick={onOpenDebug}
            className="flex items-center gap-1 text-[10px] text-gray-600 hover:text-brand-light transition-colors"
            title="RAG 全链路调试：分层观测、片段分数、Prompt、参数热调"
          >
            <Bug size={11} />
            调试
          </button>
        </div>
      </div>

      {/* 侧栏有 backdrop-filter，会成为 fixed 定位的参照系，弹窗必须挂到 body */}
      {showCreateModal &&
        createPortal(
          <CreateKBModal onClose={() => setShowCreateModal(false)} />,
          document.body,
        )}

      {/* 导出进度：后台任务 + 轮询，避免点了没反应 */}
      {exportProgress &&
        createPortal(
          <div className="fixed bottom-4 right-4 z-[80] w-[300px] rounded-2xl border border-white/[0.08] bg-surface-1 p-3.5 shadow-[0_24px_60px_-24px_#000] animate-slide-up">
            <div className="flex items-center gap-2">
              {exportProgress.status === 'running' ? (
                <Loader2 size={13} className="text-brand-light animate-spin" />
              ) : (
                <Download
                  size={13}
                  className={exportProgress.status === 'failed' ? 'text-red-400' : 'text-emerald-400'}
                />
              )}
              <span className="text-xs text-gray-200 flex-1 truncate">
                导出「{exportProgress.kbName}」
              </span>
              {exportProgress.status !== 'running' && (
                <button
                  onClick={() => setExportProgress(null)}
                  className="btn-icon p-0.5"
                  title="关闭"
                >
                  <X size={12} />
                </button>
              )}
            </div>

            <div className="mt-2.5 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
              <div
                className={clsx(
                  'h-full rounded-full transition-[width] duration-500 ease-out',
                  exportProgress.status === 'failed'
                    ? 'bg-red-500/70'
                    : 'bg-gradient-to-r from-brand to-blue-500',
                )}
                style={{ width: `${Math.max(2, Math.min(100, exportProgress.percent))}%` }}
              />
            </div>

            <div className="mt-1.5 flex items-center justify-between gap-2 text-[10px]">
              <span
                className={clsx(
                  'truncate',
                  exportProgress.status === 'failed' ? 'text-red-300' : 'text-gray-500',
                )}
              >
                {exportProgress.status === 'failed'
                  ? exportProgress.error || '导出失败'
                  : exportProgress.stage}
              </span>
              <span className="shrink-0 font-mono text-gray-600">
                {Math.round(exportProgress.percent)}%
              </span>
            </div>

            {exportProgress.status === 'running' && (
              <p className="mt-1.5 text-[10px] text-gray-700">
                向量库越大越慢，可以先做别的事，完成后自动开始下载
              </p>
            )}
            {exportProgress.status === 'done' && (
              <p className="mt-1.5 text-[10px] text-emerald-400/80">
                下载已开始，请留意浏览器下载列表
              </p>
            )}
          </div>,
          document.body,
        )}
    </aside>
  )
}
