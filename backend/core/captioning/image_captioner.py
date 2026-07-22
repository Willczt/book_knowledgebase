import structlog
from typing import Optional

from core.llm.base import LLMProvider

logger = structlog.get_logger()


class ImageCaptioner:
    """
    调用 Vision LLM 为图片生成文字描述（Caption）。
    Caption 作为文本 chunk 进入向量索引，image_path 存入 payload。
    """

    DEFAULT_PROMPT = (
        "请用中文详细描述这张图片的内容。如果是图表，请说明类型、坐标轴、数据趋势和关键数值。"
        "如果是流程图或架构图，请描述各组件和箭头关系。如果是截图或实物，请描述主要内容。"
        "描述要足够详细，以便通过文字搜索能找到这张图片。"
    )

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self._llm = llm_provider

    async def caption(self, image_bytes: bytes, page_num: Optional[int] = None) -> str:
        if self._llm is None:
            return f"[图片{f'（第{page_num}页）' if page_num else ''}，未配置视觉模型，跳过描述]"

        try:
            caption = await self._llm.caption_image(image_bytes, self.DEFAULT_PROMPT)
            return caption.strip() if caption else ""
        except NotImplementedError:
            logger.warning("caption_image_not_supported", provider=type(self._llm).__name__)
            return ""
        except Exception as e:
            logger.error("caption_image_failed", error=str(e), page_num=page_num)
            return ""
