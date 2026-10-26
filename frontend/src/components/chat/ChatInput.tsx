import { useState, useRef } from 'react'
import { Send, Image, X, Sparkles } from 'lucide-react'
import clsx from 'clsx'

import { RetrievalSettings, type RetrievalOptions } from './RetrievalSettings'

interface ChatInputProps {
  onSend: (content: string, image?: string) => void
  disabled?: boolean
  /** 检索参数（状态由父组件持有并持久化） */
  retrieval: RetrievalOptions
  onRetrievalChange: (next: RetrievalOptions) => void
}

export function ChatInput({
  onSend,
  disabled,
  retrieval,
  onRetrievalChange,
}: ChatInputProps) {
  const [content, setContent] = useState('')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [isFocused, setIsFocused] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    if (!content.trim() && !imageBase64) return
    onSend(content.trim(), imageBase64 || undefined)
    setContent('')
    setImagePreview(null)
    setImageBase64(null)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      setImagePreview(result)
      setImageBase64(result.split(',')[1])
    }
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`
  }

  const canSend = (content.trim() || imageBase64) && !disabled

  return (
    <div className="px-4 pb-4 pt-2">
      <div className={clsx('composer rounded-2xl', isFocused && 'is-focused')}>
        {/* Image Preview */}
        {imagePreview && (
          <div className="px-4 pt-3">
            <div className="relative inline-block">
              <img
                src={imagePreview}
                alt="附件"
                className="h-16 w-16 object-cover rounded-xl border border-white/10"
              />
              <button
                onClick={() => { setImagePreview(null); setImageBase64(null) }}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-surface-3 border border-white/10 rounded-full flex items-center justify-center hover:bg-red-500/80 transition-colors"
              >
                <X size={10} />
              </button>
            </div>
          </div>
        )}

        {/* Text Area */}
        <div className="flex items-end gap-2 px-4 py-3">
          <textarea
            ref={textareaRef}
            value={content}
            onChange={handleTextareaInput}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="提问，或上传图片一起检索..."
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-600
                       resize-none outline-none min-h-[22px] max-h-[180px] leading-[1.6]"
          />

          <div className="flex items-center gap-1 shrink-0 pb-0.5">
            <button
              type="button"
              onClick={() => onRetrievalChange({ ...retrieval, use_reranker: !retrieval.use_reranker })}
              aria-pressed={retrieval.use_reranker}
              className={clsx(
                'h-7 px-2 rounded-lg flex items-center gap-1 text-[11px] border transition-all duration-200',
                retrieval.use_reranker
                  ? 'text-brand-light border-brand/40 bg-brand/10 hover:bg-brand/20'
                  : 'text-gray-500 border-white/[0.07] bg-white/[0.02] hover:text-gray-300 hover:border-white/15',
              )}
              title={
                retrieval.use_reranker
                  ? '重排已开启：检索结果会经 rerank 模型精排（点击关闭）'
                  : '重排已关闭：直接使用向量检索顺序（点击开启）'
              }
            >
              <Sparkles size={13} />
              重排
            </button>

            <RetrievalSettings
              value={retrieval}
              onChange={onRetrievalChange}
              disabled={disabled}
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-icon"
              title="附加图片"
            >
              <Image size={17} />
            </button>

            <button
              onClick={handleSend}
              disabled={!canSend}
              className={clsx(
                'w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-200',
                canSend
                  ? 'btn-send active:translate-y-0'
                  : 'bg-surface-3 text-gray-600 cursor-not-allowed',
              )}
            >
              <Send size={15} />
            </button>
          </div>
        </div>
      </div>

      <p className="text-center text-[11px] text-gray-600 mt-2">
        Enter 发送 · Shift+Enter 换行
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  )
}
