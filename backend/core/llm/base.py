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
