"""本地模型失败时切到在线模型：FallbackLLM 与包装规则。"""

import pytest

from core.llm import registry
from core.llm.base import ChatMessage, LLMConfig, LLMProvider, MessageRole
from core.llm.fallback import FallbackLLM


class _FakeProvider(LLMProvider):
    def __init__(
        self,
        provider: str = "ollama",
        model_id: str = "qwen3:8b",
        reply: str = "本地回答",
        chunks: tuple[str, ...] = ("本地", "回答"),
        chat_error: Exception | None = None,
        stream_error: Exception | None = None,
        fail_after_chunk: bool = False,
        caption_error: Exception | None = None,
    ) -> None:
        super().__init__(LLMConfig(provider=provider, model_id=model_id))
        self.reply = reply
        self.chunks = chunks
        self.chat_error = chat_error
        self.stream_error = stream_error
        self.fail_after_chunk = fail_after_chunk
        self.caption_error = caption_error
        self.stream_calls = 0

    async def chat(self, messages):
        if self.chat_error is not None:
            raise self.chat_error
        return self.reply

    async def stream_chat(self, messages):
        self.stream_calls += 1
        for index, chunk in enumerate(self.chunks):
            if self.stream_error is not None and (not self.fail_after_chunk or index > 0):
                raise self.stream_error
            yield chunk
        if self.stream_error is not None and not self.fail_after_chunk:
            raise self.stream_error

    async def caption_image(self, image_bytes, prompt=""):
        if self.caption_error is not None:
            raise self.caption_error
        return "本地 caption"


_MESSAGES = [ChatMessage(role=MessageRole.user, content="你好")]


def _pair(**primary_kwargs) -> tuple[FallbackLLM, _FakeProvider, _FakeProvider]:
    primary = _FakeProvider(**primary_kwargs)
    backup = _FakeProvider(provider="deepseek", model_id="deepseek-flash", reply="在线回答", chunks=("在线", "回答"))
    return FallbackLLM(primary, backup), primary, backup


async def test_chat_falls_back_on_error() -> None:
    llm, _, _ = _pair(chat_error=ConnectionError("connection refused"))

    assert await llm.chat(_MESSAGES) == "在线回答"
    assert llm.used_fallback is True
    assert "ConnectionError" in (llm.fallback_reason or "")
    assert llm.fallback_provider == "deepseek"


async def test_chat_prefers_primary() -> None:
    llm, _, _ = _pair()

    assert await llm.chat(_MESSAGES) == "本地回答"
    assert llm.used_fallback is False
    assert llm.fallback_reason is None


async def test_stream_falls_back_before_first_token() -> None:
    error = ConnectionError("model not found")
    llm, _, _ = _pair(chunks=(), stream_error=error)

    chunks = [chunk async for chunk in llm.stream_chat(_MESSAGES)]

    assert chunks == ["在线", "回答"]
    assert llm.used_fallback is True


async def test_stream_does_not_switch_after_tokens() -> None:
    """已经吐过 token 就不再切换，避免半截回答接另一个模型的续写"""
    llm, _, backup = _pair(stream_error=RuntimeError("stream broken"), fail_after_chunk=True)
    received = []

    with pytest.raises(RuntimeError, match="stream broken"):
        async for chunk in llm.stream_chat(_MESSAGES):
            received.append(chunk)

    assert received == ["本地"]
    assert backup.stream_calls == 0
    assert llm.used_fallback is False


async def test_caption_unsupported_is_not_swallowed() -> None:
    llm, _, _ = _pair(caption_error=NotImplementedError("no vision"))

    with pytest.raises(NotImplementedError):
        await llm.caption_image(b"img")


def test_providers_list_puts_local_first() -> None:
    providers = registry.get_available_llm_providers()

    assert providers[0].provider == "ollama"


def test_maybe_wrap_fallback_wraps_local(monkeypatch) -> None:
    primary = _FakeProvider()
    backup = _FakeProvider(provider="deepseek", model_id="deepseek-flash")
    created: list[tuple[str, str]] = []

    def fake_create(provider, model_id, llm_config=None):
        created.append((provider, model_id))
        return backup

    monkeypatch.setattr(registry.settings, "llm_fallback_enabled", True)
    monkeypatch.setattr(registry.settings, "llm_fallback_provider", "deepseek")
    monkeypatch.setattr(registry.settings, "llm_fallback_model", "deepseek-flash")
    monkeypatch.setattr(registry, "create_llm_provider", fake_create)

    wrapped = registry.maybe_wrap_fallback(primary)

    assert isinstance(wrapped, FallbackLLM)
    assert created == [("deepseek", "deepseek-flash")]


def test_maybe_wrap_fallback_skips_online_primary(monkeypatch) -> None:
    primary = _FakeProvider(provider="deepseek", model_id="deepseek-flash")
    monkeypatch.setattr(registry.settings, "llm_fallback_enabled", True)

    assert registry.maybe_wrap_fallback(primary) is primary


def test_maybe_wrap_fallback_respects_switch(monkeypatch) -> None:
    primary = _FakeProvider()
    monkeypatch.setattr(registry.settings, "llm_fallback_enabled", False)

    assert registry.maybe_wrap_fallback(primary) is primary


def test_maybe_wrap_fallback_degrades_when_unconfigured(monkeypatch) -> None:
    primary = _FakeProvider()

    def broken(provider, model_id, llm_config=None):
        raise RuntimeError("DEEPSEEK_API_KEY not configured")

    monkeypatch.setattr(registry.settings, "llm_fallback_enabled", True)
    monkeypatch.setattr(registry.settings, "llm_fallback_provider", "deepseek")
    monkeypatch.setattr(registry.settings, "llm_fallback_model", None)
    monkeypatch.setattr(registry.settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(registry, "create_llm_provider", broken)

    assert registry.maybe_wrap_fallback(primary) is primary
