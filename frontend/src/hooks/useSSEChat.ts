import { useRef, useCallback } from 'react'
import type { SourceChunk } from '../types/api'

interface UseSSEChatOptions {
  onToken: (token: string) => void
  onDone: (messageId: string, sources: SourceChunk[]) => void
  onError: (error: string) => void
  /** 流在没有收到 done/error 的情况下结束（连接中断、服务重启等） */
  onIncomplete?: (reason: string) => void
}

export function useSSEChat() {
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(
    async (
      sessionId: string,
      content: string,
      image: string | undefined,
      options: UseSSEChatOptions,
    ) => {
      if (abortRef.current) {
        abortRef.current.abort()
      }
      abortRef.current = new AbortController()

      try {
        const response = await fetch(`/api/v1/sessions/${sessionId}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content,
            image: image || undefined,
          }),
          signal: abortRef.current.signal,
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          options.onError(errorData.detail || `HTTP ${response.status}`)
          return
        }

        const reader = response.body?.getReader()
        if (!reader) {
          options.onError('No response body')
          return
        }

        const decoder = new TextDecoder()
        let buffer = ''
        let doneReceived = false
        let errorReceived = false

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                if (data.token !== undefined) {
                  options.onToken(data.token)
                } else if (data.done) {
                  doneReceived = true
                  options.onDone(data.message_id, data.sources || [])
                } else if (data.error) {
                  errorReceived = true
                  options.onError(data.error)
                }
              } catch {
                // 忽略解析错误
              }
            }
          }
        }

        // 服务端异常/重启时流会直接结束，不会发 done，界面需要恢复可交互
        if (!doneReceived && !errorReceived) {
          options.onIncomplete?.('连接中断，回答可能不完整')
        }
      } catch (error: unknown) {
        if ((error as Error).name !== 'AbortError') {
          options.onError((error as Error).message || '连接失败')
        }
      }
    },
    [],
  )

  const abort = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  return { sendMessage, abort }
}
