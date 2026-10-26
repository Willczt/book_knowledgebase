import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Upload } from 'lucide-react'
import clsx from 'clsx'

import { documentsApi } from '../../api/documents'
import type { Document, IngestProgress } from '../../types/api'

export function DocumentUploader({ kbId }: { kbId: string }) {
  const [uploading, setUploading] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const queryClient = useQueryClient()

  const handleFiles = async (files: FileList) => {
    setUploading(true)
    for (const file of Array.from(files)) {
      try {
        const doc = await documentsApi.upload(kbId, file)
        const poll = async () => {
          const status = await documentsApi.status(kbId, doc.id)
          if (status.status === 'indexed' || status.status === 'failed') {
            queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
          } else {
            setTimeout(poll, 2000)
          }
        }
        poll()
      } catch (e) {
        console.error(e)
      }
    }
    setUploading(false)
    queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
  }

  return (
    <label
      className={clsx(
        'flex flex-col items-center justify-center rounded-xl py-4 px-3 cursor-pointer transition-all duration-200 border-2 border-dashed',
        isDragOver
          ? 'border-brand/60 bg-brand/10'
          : 'border-white/[0.08] hover:border-brand/40 hover:bg-brand/5',
      )}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setIsDragOver(false)
        if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files)
      }}
    >
      <Upload size={16} className={clsx('mb-1.5 transition-colors', isDragOver ? 'text-brand-light' : 'text-gray-600')} />
      <span className="text-[11px] text-gray-600 text-center">
        {uploading ? (
          <span className="text-brand-light animate-pulse">上传中...</span>
        ) : (
          <>点击或拖拽文件<br /><span className="text-gray-700">PDF · EPUB · Markdown · TXT</span></>
        )}
      </span>
      <input
        type="file"
        accept=".pdf,.epub,.md,.markdown,.txt,.text,text/plain"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && handleFiles(e.target.files)}
        disabled={uploading}
      />
    </label>
  )
}

export function useIngestProgress(
  kbId: string | null,
  docs: Document[] | undefined,
  onFinished: () => void,
): Record<string, IngestProgress> {
  const [progressMap, setProgressMap] = useState<Record<string, IngestProgress>>({})
  const finishedRef = useRef(false)

  const pendingIds = (docs ?? [])
    .filter((d) => d.status === 'pending' || d.status === 'processing')
    .map((d) => d.id)
  const pendingKey = pendingIds.join(',')

  useEffect(() => {
    if (!kbId || !pendingKey) {
      setProgressMap({})
      return
    }
    const ids = pendingKey.split(',')
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    finishedRef.current = false

    const tick = async () => {
      const next: Record<string, IngestProgress> = {}
      let allDone = true
      await Promise.all(
        ids.map(async (id) => {
          try {
            const status = await documentsApi.status(kbId, id)
            if (status.progress) next[id] = status.progress
            if (status.status !== 'indexed' && status.status !== 'failed') allDone = false
          } catch {
            allDone = false
          }
        }),
      )
      if (cancelled) return
      setProgressMap(next)
      if (allDone) {
        if (!finishedRef.current) {
          finishedRef.current = true
          onFinished()
        }
        return
      }
      timer = setTimeout(tick, 1500)
    }

    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbId, pendingKey])

  return progressMap
}
