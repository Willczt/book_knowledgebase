import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronDown, ChevronUp, FileText, Image } from 'lucide-react'
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
    <div className="mt-2 text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-gray-500 hover:text-gray-300 transition-colors"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        <span>参考来源 ({sources.length} 条)</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <div
              key={s.chunk_id}
              className="bg-gray-800 rounded-lg p-2.5 border border-gray-700"
            >
              <div className="flex items-center gap-1.5 mb-1.5">
                {s.chunk_type === 'image_caption' ? (
                  <Image size={10} className="text-blue-400" />
                ) : (
                  <FileText size={10} className="text-gray-400" />
                )}
                <span className="text-gray-400 font-medium truncate">{s.source_doc}</span>
                {s.page_num && <span className="text-gray-600">第 {s.page_num} 页</span>}
                <span className="ml-auto text-gray-600">相关度 {(s.score * 100).toFixed(0)}%</span>
              </div>
              {s.image_path && (
                <img
                  src={`/images/${s.image_path.replace('images/', '')}`}
                  alt="相关图片"
                  className="max-h-40 rounded-lg mb-1.5 border border-gray-700"
                />
              )}
              <p className="text-gray-400 text-xs line-clamp-3">{s.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={clsx('flex gap-3 mb-6', isUser && 'flex-row-reverse')}>
      {/* 头像 */}
      <div
        className={clsx(
          'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0',
          isUser ? 'bg-brand text-white' : 'bg-gray-700 text-gray-200',
        )}
      >
        {isUser ? '我' : 'AI'}
      </div>

      <div className={clsx('flex flex-col gap-1 max-w-[80%]', isUser && 'items-end')}>
        {/* 图片 */}
        {message.image_paths?.map((path, i) => (
          <img
            key={i}
            src={path.startsWith('data:') ? path : `/images/${path}`}
            alt="用户图片"
            className="max-h-48 rounded-xl border border-gray-700 mb-1"
          />
        ))}

        {/* 文字内容 */}
        <div
          className={clsx(
            'rounded-2xl px-4 py-3 text-sm',
            isUser
              ? 'bg-brand text-white rounded-tr-sm'
              : 'bg-gray-800 text-gray-200 rounded-tl-sm',
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className={clsx('markdown-body', isStreaming && !message.content && 'typing-cursor')}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content || ''}
              </ReactMarkdown>
              {isStreaming && message.content && (
                <span className="typing-cursor" />
              )}
            </div>
          )}
        </div>

        {/* 来源 */}
        {!isUser && message.sources && message.sources.length > 0 && !isStreaming && (
          <div className="px-1">
            <SourcePanel sources={message.sources} />
          </div>
        )}
      </div>
    </div>
  )
}
