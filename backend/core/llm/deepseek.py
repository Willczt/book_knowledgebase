import base64
from typing import AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import get_settings
from core.error_mapping import friendly_llm_error, is_retryable
from core.exceptions import LLMEmptyResponse
from core.llm.base import ChatMessage, LLMConfig, LLMProvider

settings = get_settings()

_CAPTION_PROMPT = (
    "请用中文详细描述这张图片的内容。如果是图表，请说明类型、坐标轴、数据趋势和关键数值。"
    "如果是流程图或架构图，请描述各组件和箭头关系。如果是截图或实物，请描述主要内容。"
    "描述要足够详细，以便通过文字搜索能找到这张图片。"
)


class DeepSeekLLM(LLMProvider):
    """DeepSeek LLM。V4.1-Flash 起支持图像输入，可用于图片 caption；是否可用由 probe() 判定"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._api_key = config.api_key or settings.deepseek_api_key or ""
        self._base_url = config.base_url or settings.deepseek_base_url
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=config.timeout,
        )
        # 缓存视觉能力探测结果
        self._supports_vision: Optional[bool] = None

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        # content 可能是字符串，也可能是多模态 list[dict]，直接透传
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def _friendly(self, exc: Exception) -> Exception:
        return friendly_llm_error(
            exc,
            provider="DeepSeek",
            base_url=self._base_url,
            model_id=self.config.model_id,
            timeout=self.config.timeout,
        )

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

    def _extra_body(self) -> dict:
        """推理强度作为非标准参数透传（DeepSeek 支持 low/high/max）"""
        if self.config.reasoning_effort:
            return {"reasoning_effort": self.config.reasoning_effort}
        return {}

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
            extra_body=self._extra_body() or None,
        )
        produced = False
        finish_reason = None
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            # 推理模型（deepseek-flash 等）先输出 reasoning_content，正文在 content 里
            content = getattr(choice.delta, "content", None)
            if content:
                produced = True
                yield content

        if not produced:
            if finish_reason == "length":
                raise LLMEmptyResponse(
                    "模型的思考过程超出 max_tokens 限额被截断，没有生成正文。"
                    f"请调大 LLM_MAX_TOKENS（当前 {self.config.max_tokens}）后重试，"
                    "或改用非推理模型。"
                )
            raise LLMEmptyResponse("模型没有返回任何内容，请重试。")

    async def probe(self) -> bool:
        """
        查询 /models 元数据，确认当前模型是否支持图像输入。
        DeepSeek-V4.1-Flash（deepseek-flash）的 input_modalities 含 "image"；
        老模型 / deepseek-v4-pro 只有 "text"。
        """
        if self._supports_vision is not None:
            return self._supports_vision

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                data = response.json().get("data", [])
        except Exception:
            self._supports_vision = False
            return False

        self._supports_vision = any(
            m.get("id") == self.config.model_id
            and "image" in (m.get("input_modalities") or [])
            for m in data
        )
        return self._supports_vision

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        caption_prompt = prompt or _CAPTION_PROMPT
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": caption_prompt},
                        ],
                    }
                ],
                # 推理模型的思考也计入 max_tokens，太小会只有思考、没有描述
                max_tokens=settings.caption_max_tokens,
                stream=False,
                timeout=settings.caption_timeout,
            )
        except Exception as exc:
            raise self._friendly(exc) from exc
        return response.choices[0].message.content or ""
