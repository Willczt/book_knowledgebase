import base64
from typing import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import get_settings
from core.error_mapping import friendly_llm_error, is_retryable
from core.llm.base import ChatMessage, LLMConfig, LLMProvider

settings = get_settings()

_CAPTION_PROMPT = (
    "Please describe this image in detail in Chinese, including chart type, data trends, "
    "text annotations and key information. If it's a flowchart or architecture diagram, "
    "explain the relationships between components."
)


class OllamaLLM(LLMProvider):
    """Ollama 本地 LLM，支持 llava 等视觉模型"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = (config.base_url or settings.ollama_base_url).rstrip("/")
        # 视觉 captioning 使用的模型（可换成 qwen2.5vl / gemma3 等多模态模型）
        self._vision_model = settings.ollama_vision_model

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        for m in messages:
            if isinstance(m.content, list):
                result.append({"role": m.role.value, "content": m.content})
            else:
                result.append({"role": m.role.value, "content": m.content})
        return result

    def _friendly(self, exc: Exception) -> Exception:
        return friendly_llm_error(
            exc,
            provider="ollama",
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
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self.config.model_id,
                    "messages": self._build_messages(messages),
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async for token in self._stream_chat(messages):
                yield token
        except Exception as exc:
            raise self._friendly(exc) from exc

    async def _stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self.config.model_id,
                    "messages": self._build_messages(messages),
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        import json
                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def probe(self) -> bool:
        """检查视觉模型（llava）是否已安装在本地 Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                names = {
                    m.get("name", "") for m in response.json().get("models", [])
                }
        except Exception:
            return False

        return any(
            name == self._vision_model or name.startswith(f"{self._vision_model}:")
            for name in names
        )

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        caption_prompt = prompt or _CAPTION_PROMPT
        try:
            async with httpx.AsyncClient(timeout=settings.caption_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._vision_model,
                        "prompt": caption_prompt,
                        "images": [b64],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as exc:
            raise friendly_llm_error(
                exc,
                provider="ollama",
                base_url=self._base_url,
                model_id=self._vision_model,
                timeout=settings.caption_timeout,
            ) from exc
