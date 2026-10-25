from typing import Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import get_settings
from core.embedding.base import EmbeddingModelInfo, EmbeddingProvider
from core.error_mapping import friendly_embedding_error, is_retryable

settings = get_settings()

# 单次请求提交的文本数量。/api/embed 支持批量输入，批量越大吞吐越高
MAX_BATCH_SIZE = 64


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
        # 运行期探测：老版本 Ollama 没有 /api/embed 批量接口
        self._supports_batch: Optional[bool] = None

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_id=self._model_id,
            provider="ollama",
            vector_dim=self._vector_dim,
            supports_multimodal=False,
            max_batch_size=MAX_BATCH_SIZE,
            description=f"Ollama Embedding ({self._model_id})",
        )

    async def embed_documents(
        self,
        texts: list[str],
        images: Optional[list[Optional[bytes]]] = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        try:
            return await self._embed_documents(texts)
        except Exception as exc:
            raise friendly_embedding_error(
                exc,
                provider="ollama",
                base_url=self._base_url,
                model_id=self._model_id,
                timeout=settings.embedding_timeout,
            ) from exc

    async def embed_query(self, text: str, image: Optional[bytes] = None) -> list[float]:
        try:
            return await self._embed_query(text)
        except Exception as exc:
            raise friendly_embedding_error(
                exc,
                provider="ollama",
                base_url=self._base_url,
                model_id=self._model_id,
                timeout=settings.embedding_timeout,
            ) from exc

    @retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=settings.embedding_timeout) as client:
            if self._supports_batch is not False:
                vectors = await self._embed_batch(client, texts)
                if vectors is not None:
                    self._supports_batch = True
                    return vectors
                self._supports_batch = False

            return [await self._embed_one(client, text) for text in texts]

    @retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_query(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=settings.embedding_timeout) as client:
            return await self._embed_one(client, text)

    async def _embed_batch(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> Optional[list[list[float]]]:
        """使用 /api/embed 一次性提交整批文本；接口不可用返回 None 由调用方降级"""
        response = await client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model_id, "input": texts},
        )
        if response.status_code in (404, 405):
            # 老版本 Ollama 没有批量接口；如果是模型不存在则 400/404 带 error 字段
            payload = _safe_json(response)
            if isinstance(payload, dict) and "not found" in str(payload.get("error", "")).lower():
                response.raise_for_status()
            return None
        response.raise_for_status()
        embeddings = response.json().get("embeddings") or []
        if len(embeddings) != len(texts):
            return None
        return embeddings

    async def _embed_one(self, client: httpx.AsyncClient, text: str) -> list[float]:
        response = await client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model_id, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]


def _safe_json(response: httpx.Response):
    try:
        return response.json()
    except Exception:
        return None
