import base64
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.llm.base import ChatMessage, LLMConfig, LLMProvider, MessageRole

settings = get_settings()

_CAPTION_PROMPT = (
    "请用中文详细描述这张图片的内容，包括图表类型、数据趋势、文字标注等关键信息。"
    "如果是流程图或架构图，请说明各组件的关系。请尽量详细，以便于文字检索。"
)


class DoubaoLLM(LLMProvider):
    """豆包 LLM，使用 OpenAI 兼容接口调用火山引擎 Ark API"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=config.api_key or settings.ark_api_key or "",
            base_url=config.base_url or settings.ark_base_url,
            timeout=config.timeout,
        )

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        for m in messages:
            if isinstance(m.content, list):
                result.append({"role": m.role.value, "content": m.content})
            else:
                result.append({"role": m.role.value, "content": m.content})
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def chat(self, messages: list[ChatMessage]) -> str:
        response = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=self._build_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=self._build_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        caption_prompt = prompt or _CAPTION_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": caption_prompt},
                ],
            }
        ]
        response = await self._client.chat.completions.create(
            model=settings.doubao_vision_chat_model,
            messages=messages,
            max_tokens=512,
            stream=False,
        )
        return response.choices[0].message.content or ""
