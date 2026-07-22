import { useState, useRef } from 'react'
import { Send, Image, X } from 'lucide-react'
import clsx from 'clsx'

interface ChatInputProps {
  onSend: (content: string, image?: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
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
      <div
        className={clsx(
          'relative rounded-2xl transition-all duration-300',
          isFocused
            ? 'gradient-border-focus shadow-brand-glow'
            : 'bg-surface-2 border border-white/[0.08]',
        )}
      >
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
                  ? 'text-white shadow-brand-glow hover:shadow-brand-glow-lg hover:-translate-y-0.5 active:translate-y-0'
                  : 'bg-surface-3 text-gray-600 cursor-not-allowed',
              )}
              style={canSend ? { background: 'linear-gradient(135deg, #7c3aed, #3b82f6)' } : undefined}
            >
              <Send size={15} />
            </button>
          </div>
        </div>
      </div>

      <p className="text-center text-[11px] text-gray-700 mt-2">
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
