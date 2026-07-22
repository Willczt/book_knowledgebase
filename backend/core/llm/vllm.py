from typing import AsyncIterator

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.llm.base import ChatMessage, LLMConfig, LLMProvider

settings = get_settings()


class VLLMLLM(LLMProvider):
    """vLLM OpenAI 兼容接口"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key or settings.vllm_api_key,
            base_url=config.base_url or settings.vllm_base_url or "http://localhost:8000",
            timeout=config.timeout,
        )

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [
            {
                "role": m.role.value,
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in messages
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def chat(self, messages: list[ChatMessage]) -> str:
        response = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=self._build_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=self._build_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        raise NotImplementedError(
            "vLLM captioning depends on the deployed model. "
            "Use a multimodal model (e.g. LLaVA) and extend this class."
        )
