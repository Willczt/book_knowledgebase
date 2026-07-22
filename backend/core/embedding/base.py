from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmbeddingModelInfo:
    model_id: str
    provider: str
    vector_dim: int
    supports_multimodal: bool = False
    max_batch_size: int = 32
    description: str = ""


class EmbeddingProvider(ABC):

    @property
    @abstractmethod
    def model_info(self) -> EmbeddingModelInfo: ...

    @abstractmethod
    async def embed_documents(
        self,
        texts: list[str],
        images: Optional[list[Optional[bytes]]] = None,
    ) -> list[list[float]]:
        """批量 embed 文档 chunk（文本 + 可选图片）"""
        ...

    @abstractmethod
    async def embed_query(
        self,
        text: str,
        image: Optional[bytes] = None,
    ) -> list[float]:
        """embed 单条查询（可附带图片）"""
        ...
