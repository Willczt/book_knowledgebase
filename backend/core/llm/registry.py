from typing import Optional

import structlog

from config import get_settings
from core.exceptions import LLMProviderNotConfigured
from core.llm.base import LLMConfig, LLMProvider
from core.llm.deepseek import DeepSeekLLM
from core.llm.doubao import DoubaoLLM
from core.llm.fallback import FallbackLLM
from core.llm.ollama import OllamaLLM
from core.llm.vllm import VLLMLLM
from db.schemas import LLMProviderInfo

logger = structlog.get_logger()
settings = get_settings()

# 这些 provider 视为"本地"：失败时值得用在线模型兜底
_LOCAL_PROVIDERS = {"ollama", "vllm"}


def _fallback_model_id(provider: str) -> str:
    if settings.llm_fallback_model:
        return settings.llm_fallback_model
    return {
        "deepseek": settings.deepseek_model,
        "doubao": settings.doubao_chat_model,
        "ollama": settings.ollama_default_chat_model,
    }.get(provider, "")


def get_available_llm_providers() -> list[LLMProviderInfo]:
    """列出可用的 LLM 提供商。

    顺序就是前端「新建会话默认模型」的优先级：**本地在前、在线在后**。
    本地模型不可用时由 `maybe_wrap_fallback` 在请求内兜底到在线模型。
    """
    providers = [
        LLMProviderInfo(
            provider="ollama",
            model_id=settings.ollama_default_chat_model,
            supports_vision=True,
            description="Ollama 本地模型（默认），需要本机或局域网内的 Ollama",
        )
    ]
    if settings.vllm_base_url:
        providers.append(LLMProviderInfo(
            provider="vllm",
            model_id="custom",
            supports_vision=False,
            description="vLLM 本地部署模型",
        ))
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
            description="DeepSeek Chat，纯文字（本地模型失败时的兜底）",
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
        temperature=cfg.get("temperature", settings.llm_temperature),
        max_tokens=cfg.get("max_tokens", settings.llm_max_tokens),
        top_p=cfg.get("top_p", 0.9),
        timeout=cfg.get("timeout", settings.llm_timeout),
        reasoning_effort=cfg.get("reasoning_effort", settings.llm_reasoning_effort),
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
    """
    返回用于图片 captioning 的多模态 LLM。

    优先级：显式配置（CAPTION_PROVIDER）→ 豆包 Vision → DeepSeek（V4.1-Flash 起支持图像输入）
    → Ollama 本地视觉模型。返回的 provider 是否真的能看图由 ImageCaptioner 的 probe() 判定。
    """
    # 1. 显式配置优先
    if settings.caption_provider and settings.caption_model_id:
        try:
            return create_llm_provider(settings.caption_provider, settings.caption_model_id)
        except Exception as e:
            logger.warning(
                "caption_provider_config_invalid",
                provider=settings.caption_provider,
                model_id=settings.caption_model_id,
                error=str(e),
            )

    # 2. 豆包 Vision
    if settings.ark_api_key:
        return create_llm_provider("doubao", settings.doubao_vision_chat_model)

    # 3. DeepSeek（deepseek-flash 等多模态模型）
    if settings.deepseek_api_key:
        return create_llm_provider("deepseek", settings.deepseek_model)

    # 4. Ollama 本地视觉模型
    try:
        return create_llm_provider("ollama", settings.ollama_vision_model)
    except Exception:
        return None


def maybe_wrap_fallback(llm: LLMProvider) -> LLMProvider:
    """本地模型外面包一层在线兜底（默认 deepseek）。

    只在 provider 属于本地（ollama / vllm）、且兜底 provider 已配置时包装；
    配置不全或兜底 provider 创建失败时原样返回，不影响主链路。
    """
    if not settings.llm_fallback_enabled:
        return llm
    if llm.config.provider not in _LOCAL_PROVIDERS:
        return llm

    provider = (settings.llm_fallback_provider or "").strip().lower()
    if not provider or provider == llm.config.provider:
        return llm

    model_id = _fallback_model_id(provider)
    if not model_id:
        logger.warning("llm_fallback_model_missing", provider=provider)
        return llm

    try:
        fallback = create_llm_provider(provider, model_id)
    except Exception as exc:
        logger.warning(
            "llm_fallback_unavailable",
            provider=provider,
            error=str(exc)[:200],
        )
        return llm

    return FallbackLLM(llm, fallback)
