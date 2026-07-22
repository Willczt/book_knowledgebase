import { useState, useRef } from 'react'
import { Send, Paperclip, X, Image } from 'lucide-react'
import clsx from 'clsx'

interface ChatInputProps {
  onSend: (content: string, image?: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [content, setContent] = useState('')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageBase64, setImageBase64] = useState<string | null>(null)
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
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`
  }

  return (
    <div className="border-t border-gray-800 bg-gray-950 p-4">
      {imagePreview && (
        <div className="relative inline-block mb-3">
          <img
            src={imagePreview}
            alt="附件"
            className="h-20 w-20 object-cover rounded-lg border border-gray-700"
          />
          <button
            onClick={() => { setImagePreview(null); setImageBase64(null) }}
            className="absolute -top-1.5 -right-1.5 bg-gray-700 rounded-full p-0.5 hover:bg-gray-600"
          >
            <X size={12} />
          </button>
        </div>
      )}
      <div className="flex items-end gap-2 bg-gray-800 rounded-2xl border border-gray-700 
                      focus-within:border-brand px-4 py-3 transition-colors">
        <textarea
          ref={textareaRef}
          value={content}
          onChange={handleTextareaInput}
          onKeyDown={handleKeyDown}
          placeholder="输入问题... (Shift+Enter 换行)"
          disabled={disabled}
          rows={1}
          className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-500 
                     resize-none outline-none min-h-[24px] max-h-[200px] leading-6"
        />
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700 transition-colors"
            title="附加图片"
          >
            <Image size={18} />
          </button>
          <button
            onClick={handleSend}
            disabled={disabled || (!content.trim() && !imageBase64)}
            className={clsx(
              'p-1.5 rounded-lg transition-all',
              content.trim() || imageBase64
                ? 'bg-brand text-white hover:bg-brand-dark'
                : 'text-gray-600 cursor-not-allowed',
            )}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
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
