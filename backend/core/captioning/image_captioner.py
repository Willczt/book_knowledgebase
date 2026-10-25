import asyncio

import structlog
from typing import Optional

from core.exceptions import ProviderUnavailable
from core.llm.base import LLMProvider

logger = structlog.get_logger()


class ImageCaptioner:
    """
    调用 Vision LLM 为图片生成文字描述（Caption）。
    Caption 作为文本 chunk 进入向量索引，image_path 存入 payload。

    为避免「模型没装/不可用」时逐张图超时重试，第一次调用前会做一次轻量探测，
    探测失败后本次摄入直接跳过所有图片 caption。
    """

    DEFAULT_PROMPT = (
        "请用中文详细描述这张图片的内容。如果是图表，请说明类型、坐标轴、数据趋势和关键数值。"
        "如果是流程图或架构图，请描述各组件和箭头关系。如果是截图或实物，请描述主要内容。"
        "描述要足够详细，以便通过文字搜索能找到这张图片。"
    )

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self._llm = llm_provider
        self._available: Optional[bool] = None
        # 并发 caption 时只探测一次
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """是否已确认可用（未探测时返回 False，由 caption() 触发探测）"""
        return bool(self._available and self._llm is not None)

    async def _ensure_available(self) -> bool:
        if self._available is not None:
            return self._available

        async with self._lock:
            if self._available is not None:  # 等锁期间已被别的协程探测过
                return self._available

            provider = self._llm
            if provider is None:
                self._available = False
                return False

            try:
                self._available = bool(await provider.probe())
            except Exception as e:  # 探测本身失败也视为不可用
                logger.warning("caption_probe_failed", error=str(e))
                self._available = False

            if not self._available:
                logger.warning(
                    "caption_model_unavailable",
                    provider=type(provider).__name__,
                    hint="未检测到可用的视觉模型，本次摄入将跳过图片描述",
                )
                self._llm = None
            return self._available

    async def caption(self, image_bytes: bytes, page_num: Optional[int] = None) -> Optional[str]:
        """
        返回图片描述：
        - None：视觉模型不可用，调用方应跳过该图片（不生成占位 chunk）
        - ""：本次 caption 失败（单张图片问题）
        """
        if not await self._ensure_available():
            return None

        try:
            caption = await self._llm.caption_image(image_bytes, self.DEFAULT_PROMPT)
            return caption.strip() if caption else ""
        except NotImplementedError:
            logger.warning("caption_image_not_supported", provider=type(self._llm).__name__)
            self._llm = None
            self._available = False
            return None
        except ProviderUnavailable as e:
            # 服务整体不可用（连接断开/鉴权失败）：后续图片不再重试，避免逐张等待超时
            logger.warning(
                "caption_provider_unavailable",
                reason=str(e),
                hint="本次摄入剩余图片将跳过描述",
            )
            self._llm = None
            self._available = False
            return None
        except Exception as e:
            logger.error("caption_image_failed", error=str(e), page_num=page_num)
            return ""
