import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronDown, ChevronUp, FileText, Image, Cpu } from 'lucide-react'
import type { Message, SourceChunk } from '../../types/api'
import clsx from 'clsx'

interface MessageBubbleProps {
  message: Message
  isStreaming?: boolean
}

function SourcePanel({ sources }: { sources: SourceChunk[] }) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null

  return (
    <div className="mt-3 text-xs animate-fade-in">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.07] text-gray-500 hover:text-gray-300 transition-all duration-150 border border-white/[0.05]"
      >
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        <span className="font-medium">引用来源</span>
        <span className="ml-1 px-1.5 py-0.5 rounded-full bg-brand/20 text-brand-light text-[10px] font-semibold">
          {sources.length}
        </span>
      </button>

      {open && (
        <div className="mt-2 space-y-2 animate-slide-up">
          {sources.map((s) => (
            <div
              key={s.chunk_id}
              className="source-card bg-surface-2/90 rounded-xl p-3 border border-white/[0.06]"
            >
              <div className="flex items-center gap-1.5 mb-2">
                <div className={clsx(
                  'w-4 h-4 rounded-md flex items-center justify-center flex-shrink-0',
                  s.chunk_type === 'image_caption'
                    ? 'bg-blue-500/20 text-blue-400'
                    : 'bg-violet-500/20 text-violet-400',
                )}>
                  {s.chunk_type === 'image_caption'
                    ? <Image size={9} />
                    : <FileText size={9} />}
                </div>
                <span className="text-gray-400 font-medium truncate flex-1">{s.source_doc}</span>
                {s.page_num && (
                  <span className="text-gray-600 shrink-0">第 {s.page_num} 页</span>
                )}
                <span className="shrink-0 px-1.5 py-0.5 rounded-md bg-green-500/10 text-green-400 text-[10px] font-medium">
                  {(s.score * 100).toFixed(0)}%
                </span>
              </div>
              {s.image_path && (
                <img
                  src={`/images/${s.image_path.replace('images/', '')}`}
                  alt="相关图片"
                  className="max-h-36 rounded-lg mb-2 border border-white/[0.06] object-cover"
                />
              )}
              <p className="text-gray-500 line-clamp-2 leading-relaxed">{s.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end mb-5 animate-slide-up">
        <div className="max-w-[78%] flex flex-col items-end gap-1">
          {message.image_paths?.map((path, i) => (
            <img
              key={i}
              src={path.startsWith('data:') ? path : `/images/${path}`}
              alt="用户图片"
              className="max-h-48 rounded-xl border border-white/[0.08] object-cover"
            />
          ))}
          <div className="bubble-user px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm leading-relaxed">
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 mb-6 animate-slide-up">
      {/* AI Avatar */}
      <div className="assistant-avatar w-7 h-7 rounded-xl flex items-center justify-center shrink-0 mt-0.5">
        <Cpu size={13} className="text-brand-light" />
      </div>

      <div className="assistant-card flex-1 min-w-0">
        {/* AI message images */}
        {message.image_paths?.map((path, i) => (
          <img
            key={i}
            src={path.startsWith('data:') ? path : `/images/${path}`}
            alt="AI图片"
            className="max-h-48 rounded-xl border border-white/[0.08] mb-2"
          />
        ))}

        {/* Content */}
        <div className={clsx(
          'markdown-body',
          isStreaming && !message.content && 'typing-cursor',
        )}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content || ''}
          </ReactMarkdown>
          {isStreaming && message.content && (
            <span className="typing-cursor" />
          )}
        </div>

        {/* Sources */}
        {!isStreaming && message.sources && message.sources.length > 0 && (
          <SourcePanel sources={message.sources} />
        )}
      </div>
    </div>
  )
}
