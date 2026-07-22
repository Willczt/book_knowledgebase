from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.embedding.base import EmbeddingModelInfo, EmbeddingProvider

settings = get_settings()


class OllamaEmbedding(EmbeddingProvider):
    def __init__(
        self,
        model_id: str,
        vector_dim: int,
        base_url: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._vector_dim = vector_dim
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_id=self._model_id,
            provider="ollama",
            vector_dim=self._vector_dim,
            supports_multimodal=False,
            max_batch_size=32,
            description=f"Ollama Embedding ({self._model_id})",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_documents(
        self,
        texts: list[str],
        images: Optional[list[Optional[bytes]]] = None,
    ) -> list[list[float]]:
        results = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for text in texts:
                vec = await self._embed_one(client, text)
                results.append(vec)
        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_query(self, text: str, image: Optional[bytes] = None) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._embed_one(client, text)

    async def _embed_one(self, client: httpx.AsyncClient, text: str) -> list[float]:
        response = await client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model_id, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]
