import { useEffect, useRef, useState } from 'react'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'
import { useSSEChat } from '../../hooks/useSSEChat'
import { useSessionStore } from '../../stores/sessionStore'
import { sessionsApi } from '../../api/sessions'
import type { Message } from '../../types/api'

interface ChatWindowProps {
  sessionId: string
}

export function ChatWindow({ sessionId }: ChatWindowProps) {
  const { messages, setMessages, appendMessage, updateLastAssistantMessage } = useSessionStore()
  const [isStreaming, setIsStreaming] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const { sendMessage } = useSSEChat()

  useEffect(() => {
    sessionsApi.messages(sessionId).then(setMessages).catch(console.error)
  }, [sessionId])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async (content: string, image?: string) => {
    if (isStreaming) return

    const userMsg: Message = {
      id: `tmp-user-${Date.now()}`,
      session_id: sessionId,
      role: 'user',
      content,
      image_paths: image ? [`data:image/png;base64,${image}`] : undefined,
      created_at: new Date().toISOString(),
    }
    appendMessage(userMsg)

    const assistantMsg: Message = {
      id: `tmp-assistant-${Date.now()}`,
      session_id: sessionId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    }
    appendMessage(assistantMsg)
    setIsStreaming(true)

    let accContent = ''

    await sendMessage(sessionId, content, image, {
      onToken: (token) => {
        accContent += token
        updateLastAssistantMessage(accContent)
      },
      onDone: (_messageId, sources) => {
        useSessionStore.getState().setMessages(
          useSessionStore.getState().messages.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: accContent, sources }
              : m,
          ),
        )
        setIsStreaming(false)
      },
      onError: (error) => {
        updateLastAssistantMessage(`[错误] ${error}`)
        setIsStreaming(false)
      },
    })
  }

  return (
    <div className="flex flex-col h-full">
      {/* 消息列表 */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-6 py-6 space-y-1"
      >
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <div className="text-4xl mb-3">📚</div>
            <p className="text-base font-medium text-gray-500">开始提问</p>
            <p className="text-sm mt-1">基于知识库中的文档回答你的问题</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            isStreaming={isStreaming && idx === messages.length - 1 && msg.role === 'assistant'}
          />
        ))}
      </div>

      {/* 输入框 */}
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  )
}
