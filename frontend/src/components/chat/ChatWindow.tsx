import { useEffect, useRef, useState } from 'react'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'
import { useSSEChat } from '../../hooks/useSSEChat'
import { useSessionStore } from '../../stores/sessionStore'
import { sessionsApi } from '../../api/sessions'
import type { Message } from '../../types/api'
import { BookOpen } from 'lucide-react'

interface ChatWindowProps {
  sessionId: string
}

export function ChatWindow({ sessionId }: ChatWindowProps) {
  const { messages, setMessages, appendMessage, updateMessage } = useSessionStore()
  const [isStreaming, setIsStreaming] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const sendingRef = useRef(false)
  const { sendMessage, abort } = useSSEChat()

  useEffect(() => {
    // 切换会话时中断正在进行的流式请求，防止旧 token 污染新会话
    abort()
    setIsStreaming(false)
    sessionsApi.messages(sessionId).then(setMessages).catch(console.error)
    return () => { abort() }
  }, [sessionId])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async (content: string, image?: string) => {
    // 用 ref 而不是 state 做互斥：state 更新是异步的，快速连点两次会发出两个并发请求
    if (sendingRef.current) return
    sendingRef.current = true

    const stamp = Date.now()
    const userMsg: Message = {
      id: `tmp-user-${stamp}`,
      session_id: sessionId,
      role: 'user',
      content,
      image_paths: image ? [`data:image/png;base64,${image}`] : undefined,
      created_at: new Date().toISOString(),
    }
    appendMessage(userMsg)

    const assistantId = `tmp-assistant-${stamp}`
    const assistantMsg: Message = {
      id: assistantId,
      session_id: sessionId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    }
    appendMessage(assistantMsg)
    setIsStreaming(true)

    let accContent = ''

    const finalize = () => {
      sendingRef.current = false
      setIsStreaming(false)
    }

    await sendMessage(sessionId, content, image, {
      onToken: (token) => {
        accContent += token
        updateMessage(assistantId, { content: accContent })
      },
      onDone: (_messageId, sources) => {
        updateMessage(assistantId, {
          content: accContent || '（模型没有返回内容，请重试）',
          sources,
        })
        finalize()
      },
      onError: (error) => {
        updateMessage(assistantId, {
          content: accContent ? `${accContent}\n\n> [出错] ${error}` : `[出错] ${error}`,
        })
        finalize()
      },
      onIncomplete: (reason) => {
        updateMessage(assistantId, {
          content: accContent ? `${accContent}\n\n> [中断] ${reason}` : `[中断] ${reason}`,
        })
        finalize()
      },
    })

    finalize()
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Messages */}
      <div
        ref={scrollRef}
        className="chat-surface flex-1 overflow-y-auto"
      >
        {messages.length === 0 ? (
          <div className="relative isolate flex flex-col items-center justify-center h-full gap-4 px-6 pb-16 animate-fade-in">
            {/* 空状态：一团柔光把内容从背景里“托”起来 */}
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 -z-10"
              style={{
                background:
                  'radial-gradient(560px 340px at 50% 44%, rgba(124,58,237,0.32), transparent 70%),' +
                  'radial-gradient(900px 260px at 50% 78%, rgba(59,130,246,0.10), transparent 72%)',
              }}
            />
            {/* Icon */}
            <div className="relative mb-1">
              <div className="halo-ring" aria-hidden="true" />
              <div
                className="relative w-14 h-14 rounded-2xl flex items-center justify-center"
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(167,139,250,0.35)',
                  boxShadow:
                    'inset 0 1px 0 rgba(255,255,255,0.14), 0 0 30px -12px rgba(124,58,237,0.95)',
                }}
              >
                <BookOpen size={26} className="text-brand-light" />
              </div>
            </div>
            <div className="text-center">
              <h3 className="text-base font-semibold text-gray-200">开始向知识库提问</h3>
              <p className="text-sm text-gray-500 mt-1.5 max-w-sm leading-relaxed">
                基于你上传的文档进行语义检索，支持文字和图片输入
              </p>
            </div>
            {/* Hint chips */}
            <div className="flex flex-wrap gap-2 justify-center max-w-md">
              {['这本书的主要观点是什么？', '第一章讲了什么？', '帮我总结核心概念'].map((hint) => (
                <button
                  key={hint}
                  onClick={() => handleSend(hint)}
                  className="sheen px-3 py-1.5 rounded-xl bg-surface-2/80 border border-white/[0.07] text-xs text-gray-500 hover:text-gray-200 hover:border-brand/40 hover:bg-brand/10 hover:shadow-[0_8px_24px_-16px_rgba(124,58,237,0.9)] transition-all duration-200"
                >
                  {hint}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-6 py-6">
            {messages.map((msg, idx) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                isStreaming={isStreaming && idx === messages.length - 1 && msg.role === 'assistant'}
              />
            ))}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="composer-dock shrink-0 max-w-3xl mx-auto w-full">
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  )
}
