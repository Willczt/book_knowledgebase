from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str | list[dict]  # list[dict] 用于多模态内容


@dataclass
class LLMConfig:
    provider: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120
    # 推理强度："low" | "high" | "max"（DeepSeek 等支持），None 用服务端默认
    reasoning_effort: Optional[str] = None


class LLMProvider(ABC):

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> str:
        """阻塞式对话，返回完整响应字符串"""
        ...

    @abstractmethod
    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """流式对话，逐个 yield token"""
        ...

    @abstractmethod
    async def caption_image(self, image_bytes: bytes, prompt: str = "") -> str:
        """
        视觉模型生成图片描述。
        不支持视觉的提供商（如 DeepSeek）应 raise NotImplementedError。
        """
        ...

    async def probe(self) -> bool:
        """
        检查该提供商当前的视觉能力是否可用（只需一次轻量请求）。
        默认认为可用；不支持视觉的提供商应返回 False。
        用于图片 caption 前的一次性探测，避免逐张图超时重试。
        """
        return True
