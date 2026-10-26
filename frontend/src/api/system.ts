import client from './client'
import type { EmbeddingModelInfo, HealthResponse } from '../types/api'

export const systemApi = {
  health: () =>
    client.get<HealthResponse>('/health', { timeout: 15_000 }).then((r) => r.data),

  embeddingModels: () =>
    client.get<EmbeddingModelInfo[]>('/embedding-models').then((r) => r.data),
}
