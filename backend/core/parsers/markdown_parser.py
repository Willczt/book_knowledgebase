import io
import os
import re
from pathlib import Path
from typing import Iterator

from PIL import Image

from config import get_settings
from core.parsers.base import DocumentParser, ExtractedImage, ParsedChunk

settings = get_settings()


class MarkdownParser(DocumentParser):
    """解析 Markdown 文档：提取文字（保留标题结构）和图片引用"""

    def parse(self, file_path: str) -> Iterator[ParsedChunk]:
        filename = os.path.basename(file_path)
        base_dir = Path(file_path).parent

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 ## 或更高级标题切分章节
        sections = self._split_by_headings(content)

        for section_title, section_text in sections:
            # 提取图片引用 ![alt](path)
            images = self._extract_images(section_text, base_dir)

            # 去除图片标记，清理纯文字
            clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", section_text)
            clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean_text)
            clean_text = re.sub(r"`{1,3}[^`]*`{1,3}", "", clean_text, flags=re.DOTALL)
            clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

            if clean_text or images:
                yield ParsedChunk(
                    text=clean_text,
                    images=images,
                    section_title=section_title,
                    source_doc=filename,
                )

    def _split_by_headings(self, content: str) -> list[tuple[str, str]]:
        """按一、二级标题切分，返回 (title, text) 列表"""
        heading_pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(content))

        if not matches:
            return [("", content)]

        sections = []
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[start:end]
            sections.append((title, section_text))

        return sections

    def _extract_images(self, text: str, base_dir: Path) -> list[ExtractedImage]:
        images = []
        pattern = re.compile(r"!\[.*?\]\(([^)]+)\)")
        for match in pattern.finditer(text):
            img_path_str = match.group(1).strip()
            # 跳过远程 URL
            if img_path_str.startswith(("http://", "https://")):
                continue

            img_path = base_dir / img_path_str
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path)
                w, h = img.size
                if w < 50 or h < 50:
                    continue

                max_dim = settings.max_image_dimension
                if max(w, h) > max_dim:
                    ratio = max_dim / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                images.append(ExtractedImage(data=buf.getvalue(), ext="png"))
            except Exception:
                continue

        return images

    @classmethod
    def supports(cls, _file_path: str) -> bool:
        return _file_path.lower().endswith((".md", ".markdown"))
