import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  Activity,
  Bug,
  Download,
  Eraser,
  Play,
  RefreshCw,
  SlidersHorizontal,
  X,
} from 'lucide-react'

import { debugApi } from '../../api/debug'
import { useKBStore } from '../../stores/kbStore'
import type { TraceSummary } from '../../types/debug'
import { TraceDetail } from './TraceDetail'

interface DebugPanelProps {
  onClose: () => void
}

const timeOf = (value: string) => value.slice(11, 19)

/** 兼容 query 为字符串或对象两种返回格式 */
const queryLabel = (value: unknown) => {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    return String(record.original ?? record.rewritten ?? '')
  }
  return ''
}

const downloadJson = (filename: string, data: unknown) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function DebugPanel({ onClose }: DebugPanelProps) {
  const queryClient = useQueryClient()
  const activeKB = useKBStore((s) => s.activeKB)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [queryText, setQueryText] = useState('')
  const [topK, setTopK] = useState(8)
  const [threshold, setThreshold] = useState(0)
  const [minRerankScore, setMinRerankScore] = useState(0)
  const [useReranker, setUseReranker] = useState(true)
  const [rerankCandidates, setRerankCandidates] = useState(30)

  const statusQuery = useQuery({ queryKey: ['debug-status'], queryFn: debugApi.status })
  const status = statusQuery.data

  const tracesQuery = useQuery({
    queryKey: ['debug-traces'],
    queryFn: () => debugApi.traces(50),
    refetchInterval: status?.enabled ? 5000 : false,
  })

  const metricsQuery = useQuery({
    queryKey: ['debug-metrics'],
    queryFn: () => debugApi.metrics(100),
    refetchInterval: status?.enabled ? 15000 : false,
  })

  const detailQuery = useQuery({
    queryKey: ['debug-trace', selectedId],
    queryFn: () => debugApi.trace(selectedId as string),
    enabled: Boolean(selectedId),
  })

  const traces = useMemo(() => tracesQuery.data || [], [tracesQuery.data])
  useEffect(() => {
    if (!selectedId && traces.length > 0) setSelectedId(traces[0].trace_id)
  }, [traces, selectedId])

  const statusMutation = useMutation({
    mutationFn: debugApi.updateStatus,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['debug-status'] })
    },
  })

  const clearMutation = useMutation({
    mutationFn: debugApi.clear,
    onSuccess: () => {
      setSelectedId(null)
      queryClient.invalidateQueries({ queryKey: ['debug-traces'] })
      queryClient.invalidateQueries({ queryKey: ['debug-status'] })
      queryClient.invalidateQueries({ queryKey: ['debug-metrics'] })
    },
  })

  const searchMutation = useMutation({
    mutationFn: () =>
      debugApi.search(activeKB?.id as string, {
        query: queryText.trim(),
        top_k: topK,
        score_threshold: threshold,
        min_rerank_score: minRerankScore,
        use_reranker: useReranker,
        rerank_candidates: rerankCandidates,
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['debug-traces'] })
      queryClient.invalidateQueries({ queryKey: ['debug-metrics'] })
      queryClient.invalidateQueries({ queryKey: ['debug-status'] })
      if (data.trace_id) setSelectedId(data.trace_id)
    },
  })

  const toggleLayer = (layer: string) => {
    if (!status) return
    const next = status.layers.includes(layer)
      ? status.layers.filter((item) => item !== layer)
      : [...status.layers, layer]
    statusMutation.mutate({ layers: next })
  }

  const exportTrace = () => {
    if (detailQuery.data) {
      downloadJson(`trace-${detailQuery.data.trace_id}.json`, detailQuery.data)
    }
  }

  return (
    <div className="fixed inset-0 z-[70] flex justify-end bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="h-full w-full max-w-6xl bg-surface-1 border-l border-white/10 flex flex-col animate-slide-up">
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-white/[0.06] shrink-0">
          <div className="logo-mark w-7 h-7 rounded-lg flex items-center justify-center">
            <Bug size={13} className="text-brand-light" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-gray-100">RAG 调试（全链路观测）</div>
            <div className="text-[10px] text-gray-600">
              {status?.enabled
                ? `已开启 · ${status.layers.length} 层 · 本机保留 ${status.trace_count} 条`
                : '已关闭：不采集新请求，可回看历史 trace'}
            </div>
          </div>
          <button
            type="button"
            disabled={!status || statusMutation.isPending}
            onClick={() => statusMutation.mutate({ enabled: !status?.enabled })}
            className={clsx(
              'h-7 px-2.5 rounded-lg text-[11px] border transition-colors disabled:opacity-40',
              status?.enabled
                ? 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10'
                : 'text-gray-400 border-white/[0.08] bg-white/[0.03]',
            )}
          >
            {status?.enabled ? '调试已开启' : '调试已关闭'}
          </button>
          <button type="button" onClick={onClose} className="btn-icon" title="关闭">
            <X size={15} />
          </button>
        </div>

        <div className="flex-1 min-h-0 flex">
          {/* Left column */}
          <div className="w-[290px] shrink-0 border-r border-white/[0.06] flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {/* Layers */}
              <div>
                <div className="text-[10px] text-gray-600 mb-1.5">观测层（可单独开关）</div>
                <div className="flex flex-wrap gap-1">
                  {(status?.available_layers || []).map((layer) => {
                    const on = status?.layers.includes(layer.name)
                    return (
                      <button
                        key={layer.name}
                        type="button"
                        onClick={() => toggleLayer(layer.name)}
                        className={clsx(
                          'px-1.5 py-1 rounded-md text-[10px] border transition-colors',
                          on
                            ? 'text-brand-light border-brand/40 bg-brand/10'
                            : 'text-gray-500 border-white/[0.07] bg-white/[0.02]',
                        )}
                      >
                        {layer.label}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Hot tuning */}
              <div className="rounded-xl border border-white/[0.06] p-2.5 space-y-2">
                <div className="flex items-center gap-1 text-[10px] text-gray-400">
                  <SlidersHorizontal size={11} /> 参数热调 / 重跑检索
                </div>
                <textarea
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  rows={2}
                  placeholder={activeKB ? '输入问题，用当前参数重跑检索' : '先在左侧选择知识库'}
                  className="input-base w-full text-[11px]"
                />
                <div className="grid grid-cols-2 gap-1.5">
                  <label className="text-[10px] text-gray-600">
                    Top-K
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="input-base w-full text-[11px] py-1 mt-0.5"
                    />
                  </label>
                  <label className="text-[10px] text-gray-600">
                    召回阈值
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={threshold}
                      onChange={(e) => setThreshold(Number(e.target.value))}
                      className="input-base w-full text-[11px] py-1 mt-0.5"
                      title="作用于 RRF 融合分（量级 0.3~1.0，与语义相似度无关），精排前过滤"
                    />
                  </label>
                  <label className="text-[10px] text-gray-600">
                    精排阈值
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={minRerankScore}
                      onChange={(e) => setMinRerankScore(Number(e.target.value))}
                      className="input-base w-full text-[11px] py-1 mt-0.5"
                      title="作用于 rerank 分（0~1 相关度），低于此值视为不可信并丢弃；0 表示不过滤"
                    />
                  </label>
                  <label className="text-[10px] text-gray-600">
                    精排候选数
                    <input
                      type="number"
                      min={2}
                      max={200}
                      value={rerankCandidates}
                      onChange={(e) => setRerankCandidates(Number(e.target.value))}
                      className="input-base w-full text-[11px] py-1 mt-0.5"
                    />
                  </label>
                  <label className="flex items-end gap-1 text-[10px] text-gray-500 pb-1">
                    <input
                      type="checkbox"
                      checked={useReranker}
                      onChange={(e) => setUseReranker(e.target.checked)}
                      className="accent-brand-light"
                    />
                    启用 Rerank
                  </label>
                </div>
                <button
                  type="button"
                  disabled={!activeKB || !queryText.trim() || searchMutation.isPending}
                  onClick={() => searchMutation.mutate()}
                  className="btn-primary w-full text-[11px] py-1.5 flex items-center justify-center gap-1 disabled:opacity-40"
                >
                  <Play size={11} />
                  {searchMutation.isPending ? '检索中…' : '重跑检索'}
                </button>
                {searchMutation.isError && (
                  <div className="text-[10px] text-red-300 break-words">
                    {(searchMutation.error as Error).message}
                  </div>
                )}
              </div>

              {/* Metrics */}
              <div className="rounded-xl border border-white/[0.06] p-2.5">
                <div className="flex items-center gap-1 text-[10px] text-gray-400 mb-1.5">
                  <Activity size={11} /> 阶段性能（P95）
                </div>
                {(metricsQuery.data?.stages || []).slice(0, 8).map((row) => (
                  <div key={row.stage} className="flex items-center gap-2 text-[10px] text-gray-600">
                    <span className="flex-1 truncate">{row.stage}</span>
                    <span className="font-mono">{row.p50_ms} / {row.p95_ms} ms</span>
                    {row.errors > 0 && <span className="text-red-400">{row.errors}✕</span>}
                  </div>
                ))}
                {!metricsQuery.data?.stages?.length && (
                  <div className="text-[10px] text-gray-700">暂无数据</div>
                )}
              </div>

              {/* Trace list */}
              <div>
                <div className="flex items-center gap-1 mb-1.5">
                  <span className="text-[10px] text-gray-600 flex-1">最近请求</span>
                  <button
                    type="button"
                    onClick={() => tracesQuery.refetch()}
                    className="btn-icon p-0.5"
                    title="刷新"
                  >
                    <RefreshCw size={11} />
                  </button>
                  <button
                    type="button"
                    onClick={() => clearMutation.mutate()}
                    className="btn-icon p-0.5"
                    title="清空 trace"
                  >
                    <Eraser size={11} />
                  </button>
                </div>
                <div className="space-y-1">
                  {traces.map((trace: TraceSummary) => (
                    <button
                      key={trace.trace_id}
                      type="button"
                      onClick={() => setSelectedId(trace.trace_id)}
                      className={clsx(
                        'w-full text-left px-2 py-1.5 rounded-lg border transition-colors',
                        selectedId === trace.trace_id
                          ? 'border-brand/40 bg-brand/10'
                          : 'border-white/[0.05] bg-white/[0.015] hover:border-white/15',
                      )}
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-gray-600 font-mono shrink-0">
                          {timeOf(trace.started_at)}
                        </span>
                        <span className="text-[11px] text-gray-300 truncate flex-1">
                          {queryLabel(trace.query)}
                        </span>
                        {trace.errors > 0 && <span className="text-[10px] text-red-400">✕</span>}
                      </div>
                      <div className="flex items-center gap-1.5 text-[10px] text-gray-700">
                        <span>{trace.total_ms.toFixed(0)} ms</span>
                        <span>{trace.candidates} 片段</span>
                        {trace.reranked && <span className="text-brand-light/60">重排</span>}
                      </div>
                    </button>
                  ))}
                  {!traces.length && (
                    <div className="text-[10px] text-gray-700 py-4 text-center">
                      还没有 trace：开启调试后提问一次
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Detail */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex items-center gap-2 px-4 py-2 border-b border-white/[0.05] shrink-0">
              <span className="text-[10px] text-gray-600 flex-1">
                {detailQuery.data ? `链路详情 · ${detailQuery.data.trace_id}` : '选择左侧请求查看链路'}
              </span>
              <button
                type="button"
                onClick={exportTrace}
                disabled={!detailQuery.data}
                className="h-7 px-2 rounded-lg border border-white/[0.08] text-[11px] text-gray-400 flex items-center gap-1 disabled:opacity-30"
                title="导出 JSON（离线分析坏 case）"
              >
                <Download size={11} /> 导出 JSON
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {detailQuery.isLoading && <div className="text-[11px] text-gray-600">加载中…</div>}
              {detailQuery.isError && (
                <div className="text-[11px] text-red-300">
                  {(detailQuery.error as Error).message}
                </div>
              )}
              {detailQuery.data && (
                <TraceDetail
                  trace={detailQuery.data}
                  onChanged={() => {
                    queryClient.invalidateQueries({ queryKey: ['debug-trace', selectedId] })
                    queryClient.invalidateQueries({ queryKey: ['debug-traces'] })
                  }}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
