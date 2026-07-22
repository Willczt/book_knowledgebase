from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class ExtractedImage:
    data: bytes         # 原始图片字节
    ext: str            # "png" | "jpg" | "jpeg"
    page_num: Optional[int] = None
    caption: Optional[str] = None   # 由 ImageCaptioner 填充


@dataclass
class ParsedChunk:
    text: str
    images: list[ExtractedImage] = field(default_factory=list)
    page_num: Optional[int] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    source_doc: str = ""
    extra_metadata: dict = field(default_factory=dict)


class DocumentParser(ABC):

    @abstractmethod
    def parse(self, file_path: str) -> Iterator[ParsedChunk]:
        """逐 chunk 解析文档，生成 ParsedChunk（含文字和图片）"""
        ...

    @classmethod
    def supports(cls, _file_path: str) -> bool:
        """根据文件扩展名判断是否支持"""
        ...
