import { useState } from 'react'
import clsx from 'clsx'
import { AlertTriangle, ChevronDown, Clock, Quote, ThumbsDown, ThumbsUp } from 'lucide-react'

import { debugApi } from '../../api/debug'
import type {
  GenerationSection,
  QuerySection,
  RerankSection,
  RetrievalRow,
  RetrievalSection,
  TraceDetail as TraceDetailData,
  TraceStage,
} from '../../types/debug'

const fmt = (value?: number | null, digits = 4) =>
  value === null || value === undefined ? '—' : Number(value).toFixed(digits)

const ms = (value?: number) => (value === undefined ? '—' : `${value.toFixed(0)} ms`)

function Panel({
  title,
  extra,
  children,
  defaultOpen = true,
}: {
  title: string
  extra?: React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="rounded-xl border border-white/[0.06] bg-white/[0.015]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        <ChevronDown
          size={12}
          className={clsx('text-gray-600 transition-transform', !open && '-rotate-90')}
        />
        <span className="text-[11px] font-medium text-gray-300">{title}</span>
        <span className="flex-1" />
        {extra}
      </button>
      {open && <div className="px-3 pb-3 space-y-2.5">{children}</div>}
    </section>
  )
}

function Pill({
  label,
  value,
  tone = 'default',
}: {
  label: string
  value: React.ReactNode
  tone?: 'default' | 'good' | 'warn' | 'muted'
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md border text-[10px] font-mono',
        tone === 'good' && 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
        tone === 'warn' && 'border-amber-500/30 bg-amber-500/10 text-amber-300',
        tone === 'muted' && 'border-white/[0.06] bg-white/[0.02] text-gray-600',
        tone === 'default' && 'border-white/[0.08] bg-white/[0.03] text-gray-400',
      )}
    >
      <span className="font-sans text-gray-600">{label}</span>
      {value}
    </span>
  )
}

function StageTimeline({ stages }: { stages: TraceStage[] }) {
  const max = Math.max(...stages.map((s) => s.duration_ms), 1)
  return (
    <div className="space-y-1.5">
      {stages.map((stage) => (
        <div key={stage.name} className="flex items-center gap-2">
          <span className="w-32 shrink-0 text-[10px] text-gray-500 truncate">{stage.name}</span>
          <div className="flex-1 h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full',
                stage.status === 'error'
                  ? 'bg-red-500/80'
                  : 'bg-gradient-to-r from-brand to-blue-500',
              )}
              style={{ width: `${Math.max(2, (stage.duration_ms / max) * 100)}%` }}
            />
          </div>
          <span className="w-16 shrink-0 text-right text-[10px] font-mono text-gray-500">
            {ms(stage.duration_ms)}
          </span>
        </div>
      ))}
    </div>
  )
}

function QueryView({ query }: { query?: QuerySection }) {
  if (!query) return <p className="text-[11px] text-gray-600">未开启 Query 层或无数据</p>
  const rewrite = query.rewrite
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 gap-1.5">
        <div>
          <div className="text-[10px] text-gray-600 mb-0.5">原始 Query</div>
          <div className="text-xs text-gray-200 break-words">{query.original || '—'}</div>
        </div>
        {rewrite?.changed && (
          <div>
            <div className="text-[10px] text-gray-600 mb-0.5">改写后（用于检索）</div>
            <div className="text-xs text-brand-light break-words">{query.rewritten}</div>
          </div>
        )}
        <div className="flex flex-wrap gap-1.5 pt-0.5">
          <Pill label="改写" value={rewrite?.method ?? 'none'} tone={rewrite?.changed ? 'good' : 'muted'} />
          <Pill label="策略" value={query.retrieval_strategy ?? '—'} />
          {query.params?.top_k !== undefined && <Pill label="top_k" value={String(query.params.top_k)} />}
          {query.params?.score_threshold !== undefined && (
            <Pill label="阈值" value={fmt(Number(query.params.score_threshold), 2)} />
          )}
          {query.history_turns !== undefined && <Pill label="历史轮次" value={query.history_turns} />}
          {query.llm && <Pill label="LLM" value={query.llm} />}
          {query.rerank_model && <Pill label="Rerank" value={query.rerank_model} />}
        </div>
        {rewrite?.reason && (
          <div className="text-[10px] text-gray-600">改写原因：{rewrite.reason}</div>
        )}
        {query.embedding && (
          <div className="text-[10px] text-gray-600 break-words">
            Embedding：{query.embedding.provider ?? '—'} / {query.embedding.model ?? '—'} ·
            dim {query.embedding.dim ?? '—'} · norm {fmt(query.embedding.norm, 3)} · 稀疏{' '}
            {query.embedding.sparse_terms ?? 0} 个 term
            {query.embedding.has_image ? ' · 含图片' : ''}
          </div>
        )}
      </div>
    </div>
  )
}

function ChunkCard({ row }: { row: RetrievalRow }) {
  const [expanded, setExpanded] = useState(false)
  const metadata = row.metadata || {}
  const metadataText = Object.entries(metadata)
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(' · ')

  return (
    <div
      className={clsx(
        'rounded-lg border px-2.5 py-2',
        row.included
          ? 'border-white/[0.07] bg-white/[0.02]'
          : 'border-white/[0.04] bg-black/20 opacity-80',
      )}
    >
      <div className="flex items-center gap-1.5 flex-wrap">
        <span
          className={clsx(
            'w-1.5 h-1.5 rounded-full shrink-0',
            row.included ? 'bg-emerald-500/80' : 'bg-gray-700',
          )}
        />
        <span className="text-[11px] text-gray-300 truncate max-w-[42%]">
          {row.source_doc || row.chunk_id.slice(0, 8)}
        </span>
        {row.page_num ? (
          <span className="text-[10px] text-gray-600">p.{row.page_num}</span>
        ) : null}
        <span className="flex-1" />
        <Pill label="融合" value={fmt(row.score, 4)} />
        <Pill label="向量" value={fmt(row.vector_score, 4)} tone={row.vector_score ? 'default' : 'muted'} />
        <Pill label="BM25" value={fmt(row.sparse_score, 4)} tone={row.sparse_score ? 'default' : 'muted'} />
        {row.rerank_score !== null && row.rerank_score !== undefined && (
          <Pill label="重排" value={fmt(row.rerank_score, 4)} tone="good" />
        )}
      </div>

      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] text-gray-600">
        <span>#{row.chunk_id.slice(0, 8)}</span>
        <span>向量排名 {row.vector_rank ?? '—'}</span>
        <span>BM25 排名 {row.sparse_rank ?? '—'}</span>
        <span>
          最终：{row.included ? `第 ${row.final_rank ?? '—'} 条` : `未进入上下文（${row.drop_reason ?? '—'}）`}
        </span>
      </div>

      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="mt-1 w-full text-left text-[11px] leading-relaxed text-gray-500 hover:text-gray-300 break-words"
      >
        {expanded ? row.text : row.text_preview || row.text.slice(0, 200)}
        {row.text.length > 200 && (
          <span className="ml-1 text-brand-light/70">{expanded ? '收起' : '展开全文'}</span>
        )}
      </button>

      {metadataText && (
        <div className="mt-1 text-[10px] text-gray-700 break-words">{metadataText}</div>
      )}
    </div>
  )
}

function RetrievalView({ data }: { data?: RetrievalSection }) {
  if (!data) return <p className="text-[11px] text-gray-600">未开启检索层或无数据</p>
  const rows = data.results || []
  const kept = rows.filter((row) => row.included)
  const dropped = rows.filter((row) => !row.included)

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Pill label="候选" value={data.candidate_count ?? 0} />
        <Pill label="过滤" value={`${data.filtered_count ?? 0} (阈值 ${fmt(data.threshold, 2)})`} tone={(data.filtered_count ?? 0) > 0 ? 'warn' : 'muted'} />
        <Pill label="返回" value={data.returned ?? 0} tone="good" />
        <Pill label="分路" value={`dense ${data.vector_hits ?? 0} / bm25 ${data.sparse_hits ?? 0}`} />
        {data.fusion?.k !== undefined && (
          <Pill label="融合" value={`${data.fusion.method ?? 'rrf'} k=${data.fusion.k}`} />
        )}
        {data.cache && <Pill label="缓存" value={data.cache.hit ? '命中' : '未命中'} tone="muted" />}
      </div>

      {(data.warnings || []).map((warning) => (
        <div key={warning} className="flex gap-1.5 text-[10px] text-amber-300/90">
          <AlertTriangle size={11} className="shrink-0 mt-0.5" />
          <span className="break-words">{warning}</span>
        </div>
      ))}
      {Object.entries(data.errors || {}).map(([key, value]) => (
        <div key={key} className="text-[10px] text-red-300/90 break-words">
          {key}: {value}
        </div>
      ))}

      <div className="space-y-1.5">
        {kept.map((row) => (
          <ChunkCard key={row.chunk_id} row={row} />
        ))}
      </div>

      {dropped.length > 0 && (
        <div className="pt-1">
          <div className="text-[10px] text-gray-600 mb-1">未进入上下文（{dropped.length}）</div>
          <div className="space-y-1.5">
            {dropped.map((row) => (
              <ChunkCard key={`${row.chunk_id}-${row.drop_reason}`} row={row} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function RerankView({ data }: { data?: RerankSection }) {
  if (!data) return <p className="text-[11px] text-gray-600">未开启重排层或未触发重排</p>
  const rows = data.candidates_detail || []
  // 兼容旧格式：candidates 曾经被写成明细数组
  const candidatesRaw = data.candidates as unknown
  const candidateCount = Array.isArray(candidatesRaw)
    ? candidatesRaw.length
    : (data.candidates ?? 0)
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Pill label="状态" value={data.status ?? '—'} tone={data.status === 'applied' ? 'good' : 'warn'} />
        <Pill label="后端" value={data.backend ?? '—'} />
        <Pill label="模型" value={data.model ?? '—'} />
        <Pill label="候选" value={candidateCount} />
        <Pill label="耗时" value={ms(data.duration_ms)} />
        {data.top_score !== null && data.top_score !== undefined && (
          <Pill label="最高分" value={fmt(data.top_score, 4)} />
        )}
      </div>
      {data.error && <div className="text-[10px] text-red-300/90 break-words">{data.error}</div>}

      {rows.length > 0 && (
        <div className="space-y-1">
          {rows.map((row) => {
            const delta = row.rank_before - row.rank_after
            return (
              <div key={row.chunk_id} className="flex items-center gap-2 text-[10px] font-mono text-gray-500">
                <span className="text-gray-600">{row.chunk_id.slice(0, 8)}</span>
                <span>{row.rank_before} → {row.rank_after}</span>
                <span className={clsx(delta > 0 && 'text-emerald-400', delta < 0 && 'text-red-400')}>
                  {delta > 0 ? `↑${delta}` : delta < 0 ? `↓${-delta}` : '—'}
                </span>
                <span className="text-gray-600">{fmt(row.score_before, 4)} → {fmt(row.score_after, 4)}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function GenerationView({ data }: { data?: GenerationSection }) {
  if (!data) return <p className="text-[11px] text-gray-600">未开启生成层或无数据</p>
  const params = data.params || {}
  const chunks = data.context_chunks || []
  const citations = data.citations || []

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Pill label="模型" value={`${params.provider ?? '—'}/${params.model_id ?? '—'}`} />
        <Pill label="temperature" value={fmt(Number(params.temperature ?? 0), 2)} />
        <Pill label="top_p" value={fmt(Number(params.top_p ?? 0), 2)} />
        <Pill label="max_tokens" value={params.max_tokens !== undefined ? String(params.max_tokens) : '—'} />
        <Pill label="prompt tokens" value={data.tokens?.prompt_estimated ?? 0} />
        <Pill label="completion" value={data.tokens?.completion_estimated ?? 0} />
        <Pill
          label="截断"
          value={data.truncation?.applied ? data.truncation?.strategy ?? '已截断' : '未截断'}
          tone={data.truncation?.applied ? 'warn' : 'muted'}
        />
      </div>

      {data.tokens?.note && <div className="text-[10px] text-gray-700">{data.tokens.note}</div>}

      <div>
        <div className="text-[10px] text-gray-600 mb-1">
          进入上下文的片段（{chunks.length}）
        </div>
        <div className="space-y-1">
          {chunks.map((chunk) => (
            <div key={`${chunk.order}-${chunk.chunk_id}`} className="flex items-center gap-2 text-[10px] text-gray-500">
              <span className="text-gray-600 w-8 shrink-0">[{chunk.ref}]</span>
              <span className="truncate flex-1">{chunk.source_doc}{chunk.page_num ? ` p.${chunk.page_num}` : ''}</span>
              <span className="font-mono text-gray-600">{chunk.tokens} tok</span>
            </div>
          ))}
        </div>
      </div>

      {citations.length > 0 && (
        <div>
          <div className="text-[10px] text-gray-600 mb-1 flex items-center gap-1">
            <Quote size={10} /> 引用判断（回答里出现的片段原句）
          </div>
          <div className="flex flex-wrap gap-1.5">
            {citations.map((item) => (
              <Pill
                key={item.chunk_id}
                label={`[${item.ref}]`}
                value={item.cited ? '已引用' : '未引用'}
                tone={item.cited ? 'good' : 'muted'}
              />
            ))}
          </div>
        </div>
      )}

      {data.output && (
        <details className="rounded-lg border border-white/[0.06] bg-black/30">
          <summary className="cursor-pointer px-2 py-1.5 text-[10px] text-gray-500">
            生成结果全文（{data.output.length} 字）
          </summary>
          <pre className="max-h-72 overflow-auto px-2 pb-2 text-[11px] text-gray-300 whitespace-pre-wrap break-words">
            {data.output}
          </pre>
        </details>
      )}

      {data.prompt && (
        <details className="rounded-lg border border-white/[0.06] bg-black/30">
          <summary className="cursor-pointer px-2 py-1.5 text-[10px] text-gray-500">
            完整 Prompt（system + history + query，{data.prompt.messages?.length ?? 0} 条消息）
          </summary>
          <div className="max-h-96 overflow-auto px-2 pb-2 space-y-1.5">
            {(data.prompt.messages || []).map((message, index) => (
              <div key={index}>
                <div className="text-[10px] text-brand-light/70">{message.role}</div>
                <pre className="text-[10px] text-gray-400 whitespace-pre-wrap break-words">
                  {message.content}
                </pre>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

function QualityView({
  trace,
  onChanged,
}: {
  trace: TraceDetailData
  onChanged: () => void
}) {
  const [stage, setStage] = useState('retrieval')
  const [note, setNote] = useState('')
  const [relevant, setRelevant] = useState('')
  const [busy, setBusy] = useState(false)

  const ranked = (trace.retrieval?.results || [])
    .filter((row) => row.included)
    .sort((a, b) => (a.final_rank ?? 0) - (b.final_rank ?? 0))

  const submitLabel = async (verdict: 'good' | 'bad') => {
    setBusy(true)
    try {
      await debugApi.label(trace.trace_id, { verdict, stage, note: note || undefined })
      onChanged()
    } finally {
      setBusy(false)
    }
  }

  const submitEvaluate = async () => {
    const ids = relevant
      .split(/[\s,，]+/)
      .map((item) => item.trim())
      .filter(Boolean)
    setBusy(true)
    try {
      await debugApi.evaluate(trace.trace_id, {
        relevant_chunk_ids: ids,
        k: Math.max(1, ranked.length || 10),
      })
      onChanged()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-2">
      {trace.quality && (
        <div className="flex flex-wrap gap-1.5">
          <Pill label="Recall" value={fmt(trace.quality.recall, 2)} tone="good" />
          <Pill label="MRR" value={fmt(trace.quality.mrr, 3)} />
          <Pill label="NDCG" value={fmt(trace.quality.ndcg, 3)} />
          <Pill label="P@K" value={fmt(trace.quality.precision, 3)} />
          {trace.quality.citation_coverage !== undefined && (
            <Pill label="引用覆盖" value={fmt(trace.quality.citation_coverage, 2)} />
          )}
        </div>
      )}

      {trace.label && (
        <div className="text-[10px]">
          <span className={trace.label.verdict === 'good' ? 'text-emerald-300' : 'text-red-300'}>
            {trace.label.verdict === 'good' ? '标记为正常' : '标记为坏 case'}
          </span>
          {trace.label.stage && <span className="text-gray-600"> · 环节 {trace.label.stage}</span>}
          {trace.label.note && <span className="text-gray-600"> · {trace.label.note}</span>}
        </div>
      )}

      <div className="flex items-center gap-1.5 flex-wrap">
        <select
          value={stage}
          onChange={(e) => setStage(e.target.value)}
          className="input-base text-[11px] py-1"
        >
          {['query', 'retrieval', 'rerank', 'generation'].map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <button
          type="button"
          disabled={busy}
          onClick={() => submitLabel('good')}
          className="h-7 px-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-[11px] text-emerald-300 flex items-center gap-1 disabled:opacity-40"
        >
          <ThumbsUp size={11} /> 正常
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => submitLabel('bad')}
          className="h-7 px-2 rounded-lg border border-red-500/30 bg-red-500/10 text-[11px] text-red-300 flex items-center gap-1 disabled:opacity-40"
        >
          <ThumbsDown size={11} /> 坏 case
        </button>
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="归因备注（可选）"
          className="input-base text-[11px] py-1 flex-1 min-w-[140px]"
        />
      </div>

      <div className="space-y-1">
        <div className="text-[10px] text-gray-600">
          离线评估：填入标准相关片段的 chunk_id（逗号/空格分隔）
        </div>
        <textarea
          value={relevant}
          onChange={(e) => setRelevant(e.target.value)}
          rows={2}
          placeholder={ranked[0] ? `例如 ${ranked[0].chunk_id}` : 'chunk_id'}
          className="input-base w-full text-[11px] font-mono"
        />
        <button
          type="button"
          disabled={busy}
          onClick={submitEvaluate}
          className="btn-primary text-[11px] py-1 px-2 disabled:opacity-40"
        >
          计算 Recall / MRR / NDCG
        </button>
      </div>
    </div>
  )
}

export function TraceDetail({ trace, onChanged }: { trace: TraceDetailData; onChanged: () => void }) {
  const queryValue = trace.query
  const querySection =
    queryValue && typeof queryValue !== 'string' ? (queryValue as QuerySection) : undefined
  const original =
    querySection?.original ||
    (typeof queryValue === 'string' ? queryValue : '') ||
    trace.query_text ||
    ''
  const stages = trace.stages || []

  return (
    <div className="space-y-2.5">
      <div>
        <div className="text-sm text-gray-200 break-words">{original}</div>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-gray-600">
          <span className="font-mono">{trace.trace_id}</span>
          <span className="flex items-center gap-1">
            <Clock size={10} /> {ms(trace.total_ms)}
          </span>
          <span>{trace.kind}</span>
          <span>{trace.started_at.replace('T', ' ')}</span>
          {trace.errors.length > 0 && (
            <span className="text-red-300">{trace.errors.length} 个错误</span>
          )}
        </div>
      </div>

      {trace.errors.length > 0 && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/[0.06] px-2.5 py-2 space-y-1">
          {trace.errors.map((error, index) => (
            <div key={index} className="text-[10px] text-red-300 break-words">
              [{error.stage}] {error.error}
            </div>
          ))}
        </div>
      )}

      {stages.length > 0 && (
        <Panel title="阶段耗时" extra={<span className="text-[10px] text-gray-600">{stages.length} 个阶段</span>}>
          <StageTimeline stages={stages} />
        </Panel>
      )}

      <Panel title="Query 处理">
        <QueryView query={querySection} />
      </Panel>

      <Panel
        title="检索召回"
        extra={
          <span className="text-[10px] text-gray-600">
            {trace.retrieval?.results?.length ?? 0} 个片段
          </span>
        }
      >
        <RetrievalView data={trace.retrieval} />
      </Panel>

      <Panel title="重排 Rerank">
        <RerankView data={trace.rerank} />
      </Panel>

      <Panel title="生成 / Prompt" defaultOpen={false}>
        <GenerationView data={trace.generation} />
      </Panel>

      <Panel title="质量评估 / 坏 case 标注" defaultOpen={false}>
        <QualityView trace={trace} onChanged={onChanged} />
      </Panel>
    </div>
  )
}
