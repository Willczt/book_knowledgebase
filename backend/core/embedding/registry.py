from typing import Optional

from config import get_settings
from core.embedding.base import EmbeddingModelInfo, EmbeddingProvider
from core.embedding.doubao import DoubaoEmbedding
from core.embedding.ollama import OllamaEmbedding
from core.embedding.vllm import VLLMEmbedding
from core.exceptions import EmbeddingModelMismatch

settings = get_settings()

# 预定义的可选 embedding 模型（供前端展示）
AVAILABLE_EMBEDDING_MODELS: list[EmbeddingModelInfo] = [
    EmbeddingModelInfo(
        model_id=settings.doubao_text_embedding_model,
        provider="doubao",
        vector_dim=2048,
        supports_multimodal=False,
        description="豆包文本 Embedding，2048 维，需要 ARK_API_KEY",
    ),
    EmbeddingModelInfo(
        model_id=settings.doubao_vision_embedding_model,
        provider="doubao",
        vector_dim=2048,
        supports_multimodal=True,
        description="豆包多模态 Embedding（文字+图片），2048 维，需要 ARK_API_KEY",
    ),
    EmbeddingModelInfo(
        model_id="nomic-embed-text",
        provider="ollama",
        vector_dim=768,
        supports_multimodal=False,
        description="Ollama nomic-embed-text，768 维，本地运行",
    ),
    EmbeddingModelInfo(
        model_id="mxbai-embed-large",
        provider="ollama",
        vector_dim=1024,
        supports_multimodal=False,
        description="Ollama mxbai-embed-large，1024 维，本地运行",
    ),
    EmbeddingModelInfo(
        model_id="bge-m3",
        provider="ollama",
        vector_dim=1024,
        supports_multimodal=False,
        description="Ollama BGE-M3，多语言，1024 维，本地运行",
    ),
]


class EmbeddingRegistry:
    """
    管理所有 EmbeddingProvider 实例。
    提供基于 (provider, model_id) 的查找，并在 KB 创建时进行一致性校验。
    """

    def __init__(self) -> None:
        self._providers: dict[tuple[str, str], EmbeddingProvider] = {}
        self._build_defaults()

    def _build_defaults(self) -> None:
        # Doubao text embedding
        if settings.ark_api_key:
            self._providers[("doubao", settings.doubao_text_embedding_model)] = DoubaoEmbedding(
                model_id=settings.doubao_text_embedding_model
            )
            self._providers[("doubao", settings.doubao_vision_embedding_model)] = DoubaoEmbedding(
                model_id=settings.doubao_vision_embedding_model
            )

        # Ollama embeddings（注册常用模型，维度由用户在 KB 创建时指定）
        for model_id, dim in [("nomic-embed-text", 768), ("mxbai-embed-large", 1024), ("bge-m3", 1024)]:
            self._providers[("ollama", model_id)] = OllamaEmbedding(
                model_id=model_id, vector_dim=dim
            )

        # vLLM（如果配置了 base_url）
        if settings.vllm_base_url:
            self._providers[("vllm", "custom")] = VLLMEmbedding(
                model_id="custom",
                vector_dim=1536,  # 默认，用户应在创建 KB 时覆盖
            )

    def register(self, provider: EmbeddingProvider) -> None:
        key = (provider.model_info.provider, provider.model_info.model_id)
        self._providers[key] = provider

    def register_dynamic(
        self,
        provider_name: str,
        model_id: str,
        vector_dim: int,
        **kwargs,
    ) -> EmbeddingProvider:
        """按需创建并注册 provider（例如 Ollama 自定义模型）"""
        if provider_name == "ollama":
            p = OllamaEmbedding(model_id=model_id, vector_dim=vector_dim, **kwargs)
        elif provider_name == "vllm":
            p = VLLMEmbedding(model_id=model_id, vector_dim=vector_dim, **kwargs)
        elif provider_name == "doubao":
            p = DoubaoEmbedding(model_id=model_id, **kwargs)
        else:
            raise ValueError(f"Unknown embedding provider: {provider_name}")
        self.register(p)
        return p

    def get(self, provider: str, model_id: str) -> EmbeddingProvider:
        key = (provider, model_id)
        if key not in self._providers:
            # 尝试动态查找（对于用户自定义的 Ollama 或 vLLM 模型）
            raise KeyError(
                f"Embedding provider not configured: provider={provider}, model_id={model_id}. "
                "Check that the provider is running and the model is available."
            )
        return self._providers[key]

    def validate_for_kb(
        self,
        kb_embedding_provider: str,
        kb_embedding_model_id: str,
        kb_embedding_vector_dim: int,
    ) -> EmbeddingProvider:
        """校验 KB 锁定的 embedding 模型当前是否可用，维度是否匹配"""
        provider = self.get(kb_embedding_provider, kb_embedding_model_id)
        if provider.model_info.vector_dim != kb_embedding_vector_dim:
            raise EmbeddingModelMismatch(
                f"KB embedding vector dim mismatch: "
                f"expected {kb_embedding_vector_dim}, "
                f"got {provider.model_info.vector_dim} from {kb_embedding_model_id}"
            )
        return provider


_registry: Optional[EmbeddingRegistry] = None


def get_embedding_registry() -> EmbeddingRegistry:
    global _registry
    if _registry is None:
        _registry = EmbeddingRegistry()
    return _registry
