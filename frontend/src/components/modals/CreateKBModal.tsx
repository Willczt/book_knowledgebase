import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Database, X } from 'lucide-react'
import clsx from 'clsx'

import { knowledgeBaseApi } from '../../api/knowledgeBase'
import { systemApi } from '../../api/system'
import { useKBStore } from '../../stores/kbStore'
import type { EmbeddingModelInfo } from '../../types/api'

export function CreateKBModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [provider, setProvider] = useState('doubao')
  const [modelId, setModelId] = useState('doubao-embedding-vision-251215')
  const [dim, setDim] = useState(2048)
  const queryClient = useQueryClient()
  const addKB = useKBStore((s) => s.addKB)

  const { data: models } = useQuery<EmbeddingModelInfo[]>({
    queryKey: ['embedding-models'],
    queryFn: async () => {
      return systemApi.embeddingModels()
    },
  })

  // 默认选中第一个「当前真正可用」的模型（例如去掉 Ark Key 后不再默认豆包）
  useEffect(() => {
    if (!models?.length) return
    const exists = models.some((m) => m.provider === provider && m.model_id === modelId)
    if (!exists) {
      setProvider(models[0].provider)
      setModelId(models[0].model_id)
      setDim(models[0].vector_dim)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models])

  const mutation = useMutation({
    mutationFn: () =>
      knowledgeBaseApi.create({
        name,
        description: desc || undefined,
        embedding_provider: provider,
        embedding_model_id: modelId,
        embedding_vector_dim: dim,
      }),
    onSuccess: (kb) => {
      addKB(kb)
      queryClient.invalidateQueries({ queryKey: ['kbs'] })
      onClose()
    },
  })

  const handleSelectModel = (m: EmbeddingModelInfo) => {
    setProvider(m.provider)
    setModelId(m.model_id)
    setDim(m.vector_dim)
  }

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
      <div className="glass-card w-full max-w-md p-6 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="logo-mark w-7 h-7 rounded-lg flex items-center justify-center">
              <Database size={13} className="text-brand-light" />
            </div>
            <h2 className="text-sm font-semibold text-gray-100">新建知识库</h2>
          </div>
          <button onClick={onClose} className="btn-icon"><X size={15} /></button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block font-medium">名称 *</label>
            <input
              className="input-base w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="如：机器学习书单"
              autoFocus
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block font-medium">描述</label>
            <textarea
              className="input-base w-full resize-none"
              rows={2}
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="可选"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-2 block font-medium">
              Embedding 模型
              <span className="ml-1.5 text-[10px] text-amber-500/80 font-normal">创建后不可更改</span>
            </label>
            <div className="space-y-2">
              {models?.map((m) => (
                <label
                  key={`${m.provider}/${m.model_id}`}
                  className={clsx(
                    'flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-150',
                    provider === m.provider && modelId === m.model_id
                      ? 'border-brand/60 bg-brand/10'
                      : 'border-white/[0.06] hover:border-white/[0.12] bg-surface-2/60',
                  )}
                >
                  <input
                    type="radio"
                    className="mt-0.5 accent-brand"
                    checked={provider === m.provider && modelId === m.model_id}
                    onChange={() => handleSelectModel(m)}
                  />
                  <div className="text-xs">
                    <div className="font-semibold text-gray-200">{m.model_id}</div>
                    <div className="text-gray-500 mt-0.5">{m.description}</div>
                    <div className="text-gray-700 mt-0.5 font-mono">dim={m.vector_dim}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className="btn-ghost text-sm">取消</button>
          <button
            onClick={() => mutation.mutate()}
            disabled={!name.trim() || mutation.isPending}
            className="btn-primary text-sm px-5"
          >
            {mutation.isPending ? '创建中...' : '创建知识库'}
          </button>
        </div>
        {mutation.isError && (
          <p className="text-red-400 text-xs mt-2 px-1">{mutation.error.message}</p>
        )}
      </div>
    </div>
  )
}
