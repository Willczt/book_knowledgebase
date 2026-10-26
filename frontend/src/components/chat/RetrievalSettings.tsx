import { useEffect, useRef, useState } from 'react'
import { RotateCcw, Settings2, X } from 'lucide-react'
import clsx from 'clsx'

/** 一次提问的检索参数（与后端 retrieval_config 字段一致） */
export interface RetrievalOptions {
  top_k: number
  score_threshold: number
  use_reranker: boolean
}

export const RETRIEVAL_DEFAULTS: RetrievalOptions = {
  top_k: 8,
  score_threshold: 0,
  use_reranker: true,
}

export function isDefaultRetrieval(value: RetrievalOptions): boolean {
  return (
    value.top_k === RETRIEVAL_DEFAULTS.top_k &&
    value.score_threshold === RETRIEVAL_DEFAULTS.score_threshold &&
    value.use_reranker === RETRIEVAL_DEFAULTS.use_reranker
  )
}

interface RetrievalSettingsProps {
  value: RetrievalOptions
  onChange: (next: RetrievalOptions) => void
  disabled?: boolean
}

export function RetrievalSettings({ value, onChange, disabled }: RetrievalSettingsProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onPointerDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const patch = (next: Partial<RetrievalOptions>) => onChange({ ...value, ...next })
  const customized = !isDefaultRetrieval(value)

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label="检索设置"
        title="检索设置：Top-K / 相似度阈值 / Reranker"
        className={clsx(
          'h-7 px-2 rounded-lg flex items-center gap-1 text-[11px] border transition-all duration-200',
          open || customized
            ? 'text-brand-light border-brand/40 bg-brand/10 hover:bg-brand/20'
            : 'text-gray-500 border-white/[0.07] bg-white/[0.02] hover:text-gray-300 hover:border-white/15',
        )}
      >
        <Settings2 size={13} />
        设置
      </button>

      {open && (
        <div className="absolute bottom-full right-0 z-30 mb-2 w-72 rounded-2xl border border-white/[0.08] bg-surface-1 p-4 shadow-[0_24px_60px_-24px_#000] animate-slide-up">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-gray-200">检索设置</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="btn-icon p-0.5"
              title="关闭"
            >
              <X size={13} />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] text-gray-500">Top-K（召回数量）</label>
                <span className="text-[11px] font-mono text-brand-light">{value.top_k}</span>
              </div>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={value.top_k}
                disabled={disabled}
                onChange={(e) => patch({ top_k: Number(e.target.value) })}
                className="w-full h-1 cursor-pointer accent-brand-light disabled:cursor-not-allowed"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] text-gray-500">相似度阈值</label>
                <span className="text-[11px] font-mono text-brand-light">
                  {value.score_threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={value.score_threshold}
                disabled={disabled}
                onChange={(e) => patch({ score_threshold: Number(e.target.value) })}
                className="w-full h-1 cursor-pointer accent-brand-light disabled:cursor-not-allowed"
              />
              <p className="mt-1 text-[10px] text-gray-700">0 表示不过滤，数值越高越严格</p>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="text-[11px] text-gray-400">启用 Reranker</div>
                <div className="text-[10px] text-gray-700 mt-0.5">CrossEncoder 二次排序，更准但稍慢</div>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={value.use_reranker}
                aria-label="启用 Reranker"
                disabled={disabled}
                onClick={() => patch({ use_reranker: !value.use_reranker })}
                className={clsx(
                  'shrink-0 w-8 h-[18px] rounded-full flex items-center px-0.5 transition-colors duration-200 disabled:opacity-40',
                  value.use_reranker
                    ? 'bg-brand/70 border border-brand/50 justify-end'
                    : 'bg-white/[0.07] border border-white/[0.09] justify-start',
                )}
              >
                <span className="block w-3.5 h-3.5 rounded-full bg-white shadow" />
              </button>
            </div>
          </div>

          <div className="mt-3.5 pt-3 border-t border-white/[0.05] flex items-center justify-between">
            <span className="text-[10px] text-gray-700">仅作用于后续提问</span>
            <button
              type="button"
              onClick={() => onChange({ ...RETRIEVAL_DEFAULTS })}
              disabled={!customized}
              className="flex items-center gap-1 text-[11px] text-gray-500 hover:text-brand-light transition-colors disabled:opacity-35 disabled:hover:text-gray-500"
            >
              <RotateCcw size={11} />
              恢复默认
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
