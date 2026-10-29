"""LLM 兜底：本地模型不可用时自动切到在线模型。

触发条件：主 provider 在**产出第一个 token 之前**抛异常（连接被拒、模型未拉取、
请求超时等）。一旦已经吐出内容就不再切换——半截回答接另一个模型的续写会前后矛盾，
那种情况宁可把错误抛给上层。

失败信息记在实例上（`used_fallback` / `fallback_reason`），调用方可以写进 trace，
事后能判断"这次回答到底是谁生成的"。
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from core.llm.base import ChatMessage, LLMProvider

logger = structlog.get_logger()


class FallbackLLM(LLMProvider):

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        super().__init__(primary.config)
        self._primary = primary
        self._fallback = fallback
        self.used_fallback = False
        self.fallback_reason: str | None = None

    @property
    def fallback_provider(self) -> str:
        return self._fallback.config.provider

    def _switch(self, exc: Exception) -> None:
        self.used_fallback = True
        self.fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning(
            "llm_fallback",
            primary=f"{self._primary.config.provider}/{self._primary.config.model_id}",
            fallback=f"{self._fallback.config.provider}/{self._fallback.config.model_id}",
            error=self.fallback_reason,
        )

    async def chat(self, messages: list[ChatMessage]) -> str:
        try:
            return await self._primary.chat(messages)
        except Exception as exc:
            self._switch(exc)
            return await self._fallback.chat(messages)

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        started = False
        try:
            async for chunk in self._primary.stream_chat(messages):
                started = True
                yield chunk
        except Exception as exc:
            if started:
                raise
            self._switch(exc)
            async for chunk in self._fallback.stream_chat(messages):
                yield chunk

    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        try:
            return await self._primary.caption_image(image_bytes, prompt)
        except NotImplementedError:
            # 能力缺失不是故障，交给上层按"不支持视觉"处理
            raise
        except Exception as exc:
            self._switch(exc)
            return await self._fallback.caption_image(image_bytes, prompt)

    async def probe(self) -> bool:
        try:
            return await self._primary.probe()
        except Exception:
            return await self._fallback.probe()
