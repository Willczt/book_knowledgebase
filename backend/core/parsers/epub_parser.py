import io
import os
import re
from typing import Iterator

from PIL import Image

from config import get_settings
from core.parsers.base import DocumentParser, ExtractedImage, ParsedChunk

settings = get_settings()


class EPUBParser(DocumentParser):
    """使用 ebooklib + BeautifulSoup4 解析 EPUB"""

    MIN_TEXT_LENGTH = 30

    def parse(self, file_path: str) -> Iterator[ParsedChunk]:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(file_path, options={"ignore_ncx": True})
        filename = os.path.basename(file_path)

        # 建立图片 ID → 字节的映射
        image_map: dict[str, bytes] = {}
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                image_map[item.get_id()] = item.get_content()
                # 同时按文件名索引
                image_map[item.get_name()] = item.get_content()

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            soup = BeautifulSoup(item.get_content(), "lxml")

            # 章节标题
            chapter_title = ""
            heading = soup.find(re.compile(r"^h[1-6]$"))
            if heading:
                chapter_title = heading.get_text(strip=True)

            # 按段落切分
            paragraphs = soup.find_all(["p", "div", "section"])
            chunk_texts = []
            chunk_images: list[ExtractedImage] = []

            for para in paragraphs:
                # 提取图片引用
                for img_tag in para.find_all("img"):
                    src = img_tag.get("src", "")
                    img_data = self._resolve_image(src, image_map)
                    if img_data:
                        chunk_images.append(img_data)

                text = para.get_text(separator=" ", strip=True)
                if len(text) >= self.MIN_TEXT_LENGTH:
                    chunk_texts.append(text)

            # 处理段落级 img（在 p 外的）
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                img_data = self._resolve_image(src, image_map)
                if img_data and img_data not in chunk_images:
                    chunk_images.append(img_data)

            combined_text = "\n\n".join(chunk_texts)
            if combined_text or chunk_images:
                yield ParsedChunk(
                    text=combined_text,
                    images=chunk_images,
                    chapter=chapter_title,
                    source_doc=filename,
                )

    def _resolve_image(self, src: str, image_map: dict[str, bytes]) -> "ExtractedImage | None":
        # 尝试多种 key 形式
        keys_to_try = [
            src,
            os.path.basename(src),
            src.lstrip("./"),
        ]
        for key in keys_to_try:
            if key in image_map:
                try:
                    img_bytes = image_map[key]
                    img = Image.open(io.BytesIO(img_bytes))
                    w, h = img.size
                    if w < 50 or h < 50:
                        return None

                    # 统一转 PNG，调整大小
                    max_dim = settings.max_image_dimension
                    if max(w, h) > max_dim:
                        ratio = max_dim / max(w, h)
                        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="PNG")
                    return ExtractedImage(data=buf.getvalue(), ext="png")
                except Exception:
                    return None
        return None

    @classmethod
    def supports(cls, _file_path: str) -> bool:
        return _file_path.lower().endswith(".epub")
