from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.embedding.base import EmbeddingModelInfo, EmbeddingProvider

settings = get_settings()


class VLLMEmbedding(EmbeddingProvider):
    """使用 vLLM OpenAI 兼容 /v1/embeddings 端点"""

    def __init__(
        self,
        model_id: str,
        vector_dim: int,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._vector_dim = vector_dim
        url = base_url or settings.vllm_base_url or "http://localhost:8000"
        key = api_key or settings.vllm_api_key
        self._client = AsyncOpenAI(base_url=url, api_key=key)

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_id=self._model_id,
            provider="vllm",
            vector_dim=self._vector_dim,
            supports_multimodal=False,
            max_batch_size=32,
            description=f"vLLM Embedding ({self._model_id})",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_documents(
        self,
        texts: list[str],
        images: Optional[list[Optional[bytes]]] = None,
    ) -> list[list[float]]:
        batch_size = self.model_info.max_batch_size
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(model=self._model_id, input=batch)
            results.extend([item.embedding for item in sorted(response.data, key=lambda x: x.index)])
        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_query(self, text: str, image: Optional[bytes] = None) -> list[float]:
        response = await self._client.embeddings.create(model=self._model_id, input=[text])
        return response.data[0].embedding
