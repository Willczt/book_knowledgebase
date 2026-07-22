import base64
import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.embedding.base import EmbeddingModelInfo, EmbeddingProvider

settings = get_settings()
logger = logging.getLogger(__name__)

# 已知 Doubao embedding 模型的向量维度
_DOUBAO_EMBED_DIMS: dict[str, int] = {
    "doubao-embedding-text-240715": 2048,
    "doubao-embedding-vision-250615": 2048,
    "doubao-embedding-vision-251215": 2048,
}


class DoubaoEmbedding(EmbeddingProvider):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or settings.ark_api_key or ""
        self._base_url = base_url or settings.ark_base_url
        self._is_multimodal = "vision" in model_id

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_id=self._model_id,
            provider="doubao",
            vector_dim=_DOUBAO_EMBED_DIMS.get(self._model_id, 2048),
            supports_multimodal=self._is_multimodal,
            max_batch_size=16 if self._is_multimodal else 32,
            description=f"Doubao Embedding ({self._model_id})",
        )

    async def embed_documents(
        self,
        texts: list[str],
        images: Optional[list[Optional[bytes]]] = None,
    ) -> list[list[float]]:
        results = []
        batch_size = self.model_info.max_batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_images = images[i : i + batch_size] if images else None
            batch_result = await self._embed_batch(batch_texts, batch_images)

            # 豆包 text embedding API 批量请求只返回 1 条时，降级为逐条调用
            if len(batch_result) < len(batch_texts):
                logger.warning(
                    "doubao_embed_batch_mismatch: expected=%d got=%d, falling back to per-item",
                    len(batch_texts), len(batch_result),
                )
                batch_result = []
                for j, t in enumerate(batch_texts):
                    img_item = [batch_images[j]] if batch_images and j < len(batch_images) else None
                    single = await self._embed_batch([t], img_item)
                    batch_result.append(single[0])

            results.extend(batch_result)
        return results

    async def embed_query(self, text: str, image: Optional[bytes] = None) -> list[float]:
        result = await self._embed_batch([text], [image] if image else None)
        return result[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _embed_batch(
        self,
        texts: list[str],
        images: Optional[list[Optional[bytes]]] = None,
    ) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        if self._is_multimodal:
            # 多模态 embedding：每个 text/image 是独立 item
            # API 格式：{"type":"text","text":"..."} 或 {"type":"image_url","image_url":{"url":"..."}}
            input_items = []
            for idx, text in enumerate(texts):
                img = images[idx] if images and idx < len(images) else None
                if img:
                    b64 = base64.b64encode(img).decode()
                    input_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                else:
                    input_items.append({"type": "text", "text": text})
            payload = {"model": self._model_id, "input": input_items}
            endpoint = f"{self._base_url}/embeddings/multimodal"
        else:
            payload = {"model": self._model_id, "input": texts}
            endpoint = f"{self._base_url}/embeddings"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        # 健壮的响应解析：API 有时以 200 OK 返回字符串或带 error 字段的 dict
        if not isinstance(data, dict):
            raise ValueError(
                f"Doubao embedding returned non-dict (type={type(data).__name__}): {str(data)[:300]}"
            )

        if "error" in data:
            raise ValueError(f"Doubao embedding API error: {data['error']}")

        raw_data = data.get("data")
        if raw_data is None:
            logger.error("doubao_embed_missing_data_key: keys=%s body=%s", list(data.keys()), str(data)[:300])
            raise ValueError(f"No 'data' key in Doubao response. Keys: {list(data.keys())}")

        # 格式 A：单条 dict，如 {"embedding": [...]}（豆包文本 embedding 单条返回格式）
        if isinstance(raw_data, dict):
            if "embedding" in raw_data:
                return [raw_data["embedding"]]
            # 可能是 {"embeddings": [[...], [...]]}
            if "embeddings" in raw_data:
                return raw_data["embeddings"]
            logger.error("doubao_embed_dict_no_embedding: keys=%s", list(raw_data.keys()))
            raise ValueError(f"Unexpected dict structure in 'data': keys={list(raw_data.keys())}")

        if not isinstance(raw_data, list):
            logger.error("doubao_embed_bad_data_type: type=%s value=%s", type(raw_data).__name__, str(raw_data)[:200])
            raise ValueError(f"Expected list for 'data', got {type(raw_data).__name__}: {str(raw_data)[:200]}")

        if not raw_data:
            return []

        first = raw_data[0]
        if isinstance(first, dict):
            # OpenAI 兼容格式：[{"embedding": [...], "index": 0}, ...]
            return [item["embedding"] for item in sorted(raw_data, key=lambda x: x["index"])]
        if isinstance(first, list):
            # 简化格式：[[0.1, ...], ...]
            return raw_data

        logger.error("doubao_embed_unknown_item_type: type=%s item=%s", type(first).__name__, str(first)[:100])
        raise ValueError(f"Unexpected embedding item type {type(first).__name__}: {str(first)[:100]}")
