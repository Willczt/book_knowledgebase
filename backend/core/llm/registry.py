from typing import Optional

from config import get_settings
from core.exceptions import LLMProviderNotConfigured
from core.llm.base import LLMConfig, LLMProvider
from core.llm.doubao import DoubaoLLM
from core.llm.deepseek import DeepSeekLLM
from core.llm.ollama import OllamaLLM
from core.llm.vllm import VLLMLLM
from db.schemas import LLMProviderInfo

settings = get_settings()


def get_available_llm_providers() -> list[LLMProviderInfo]:
    providers = []
    if settings.ark_api_key:
        providers.append(LLMProviderInfo(
            provider="doubao",
            model_id=settings.doubao_chat_model,
            supports_vision=True,
            description="豆包大模型（doubao-seed），支持视觉理解",
        ))
    if settings.deepseek_api_key:
        providers.append(LLMProviderInfo(
            provider="deepseek",
            model_id=settings.deepseek_model,
            supports_vision=False,
            description="DeepSeek Chat，纯文字",
        ))
    providers.append(LLMProviderInfo(
        provider="ollama",
        model_id=settings.ollama_default_chat_model,
        supports_vision=True,
        description="Ollama 本地模型（llama3/llava），需要本地运行 Ollama",
    ))
    if settings.vllm_base_url:
        providers.append(LLMProviderInfo(
            provider="vllm",
            model_id="custom",
            supports_vision=False,
            description="vLLM 本地部署模型",
        ))
    return providers


def create_llm_provider(
    provider: str,
    model_id: str,
    llm_config: Optional[dict] = None,
) -> LLMProvider:
    """根据 provider 名称和 model_id 创建 LLMProvider 实例"""
    cfg = llm_config or {}
    config = LLMConfig(
        provider=provider,
        model_id=model_id,
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 2048),
        top_p=cfg.get("top_p", 0.9),
        timeout=cfg.get("timeout", 120),
    )

    if provider == "doubao":
        if not settings.ark_api_key:
            raise LLMProviderNotConfigured("ARK_API_KEY not configured for Doubao provider.")
        return DoubaoLLM(config)
    elif provider == "deepseek":
        if not settings.deepseek_api_key:
            raise LLMProviderNotConfigured("DEEPSEEK_API_KEY not configured.")
        return DeepSeekLLM(config)
    elif provider == "ollama":
        return OllamaLLM(config)
    elif provider == "vllm":
        if not settings.vllm_base_url:
            raise LLMProviderNotConfigured("VLLM_BASE_URL not configured.")
        return VLLMLLM(config)
    else:
        raise LLMProviderNotConfigured(f"Unknown LLM provider: {provider}")


def get_caption_provider() -> Optional[LLMProvider]:
    """返回用于图片 captioning 的 LLM，优先 Doubao，降级到 Ollama"""
    if settings.ark_api_key:
        return create_llm_provider("doubao", settings.doubao_vision_chat_model)
    try:
        return create_llm_provider("ollama", "llava")
    except Exception:
        return None
