import base64
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
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
        # 视觉 captioning 使用的模型
        self._vision_model = "llava"

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

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        caption_prompt = prompt or _CAPTION_PROMPT
        async with httpx.AsyncClient(timeout=120.0) as client:
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
