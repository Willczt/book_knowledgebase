from typing import AsyncIterator

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import get_settings
from core.error_mapping import friendly_llm_error, is_retryable
from core.llm.base import ChatMessage, LLMConfig, LLMProvider

settings = get_settings()


class VLLMLLM(LLMProvider):
    """vLLM OpenAI 兼容接口"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or settings.vllm_base_url or "http://localhost:8000"
        self._client = AsyncOpenAI(
            api_key=config.api_key or settings.vllm_api_key,
            base_url=self._base_url,
            timeout=config.timeout,
        )

    def _friendly(self, exc: Exception) -> Exception:
        return friendly_llm_error(
            exc,
            provider="vllm",
            base_url=self._base_url,
            model_id=self.config.model_id,
            timeout=self.config.timeout,
        )

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [
            {
                "role": m.role.value,
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in messages
        ]

    async def chat(self, messages: list[ChatMessage]) -> str:
        try:
            return await self._chat(messages)
        except Exception as exc:
            raise self._friendly(exc) from exc

    @retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _chat(self, messages: list[ChatMessage]) -> str:
        response = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=self._build_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async for token in self._stream_chat(messages):
                yield token
        except Exception as exc:
            raise self._friendly(exc) from exc

    async def _stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=self._build_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        raise NotImplementedError(
            "vLLM captioning depends on the deployed model. "
            "Use a multimodal model (e.g. LLaVA) and extend this class."
        )
