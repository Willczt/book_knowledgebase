import io
import os
import re
import warnings
from typing import Iterator, Optional

from PIL import Image

from config import get_settings
from core.parsers.base import DocumentParser, ExtractedImage, ParsedChunk

settings = get_settings()

# 块级标签：用于按「叶子块」抽取段落，避免容器块重复输出子块文字
_BLOCK_TAGS = (
    "p", "div", "section", "article", "li", "blockquote", "dd", "dt",
    "td", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
)


class EPUBParser(DocumentParser):
    """使用 ebooklib + BeautifulSoup4 解析 EPUB"""

    MIN_TEXT_LENGTH = 30
    MIN_IMAGE_SIZE = 50

    def parse(self, file_path: str) -> Iterator[ParsedChunk]:
        import ebooklib
        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        book = epub_read(file_path)
        filename = os.path.basename(file_path)

        # 建立图片 ID / 名称 → 字节的映射（每张图只解压一次）
        image_map: dict[str, bytes] = {}
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_IMAGE:
                continue
            data = item.get_content()
            name = item.get_name()
            image_map[item.get_id()] = data
            image_map[name] = data
            image_map[os.path.basename(name)] = data

        # 同一张图可能被多处引用：解析结果按 src 缓存，避免重复解码/重编码
        image_cache: dict[str, Optional[ExtractedImage]] = {}

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            soup = BeautifulSoup(item.get_content(), "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()

            chapter_title = self._chapter_title(soup, item)
            paragraphs = self._extract_paragraphs(soup)
            images = self._extract_images(soup, image_map, image_cache)

            combined_text = "\n\n".join(paragraphs)
            if combined_text or images:
                yield ParsedChunk(
                    text=combined_text,
                    images=images,
                    chapter=chapter_title,
                    source_doc=filename,
                )

    # ── 文本抽取 ─────────────────────────────────────────────────────
    def _chapter_title(self, soup, item) -> str:
        heading = soup.find(_HEADING_RE)
        if heading:
            title = heading.get_text(strip=True)
            if title:
                return title
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        return ""

    def _extract_paragraphs(self, soup) -> list[str]:
        blocks = soup.find_all(_BLOCK_TAGS)
        if not blocks:
            text = soup.get_text("\n", strip=True)
            return [text] if len(text) >= self.MIN_TEXT_LENGTH else []

        # 标记"含块级子节点"的容器块，跳过它们可避免文字重复
        container_ids: set[int] = set()
        for el in blocks:
            for parent in el.parents:
                container_ids.add(id(parent))

        paragraphs: list[str] = []
        for el in blocks:
            if id(el) in container_ids:
                continue
            text = el.get_text(" ", strip=True)
            if len(text) >= self.MIN_TEXT_LENGTH:
                paragraphs.append(text)
        return paragraphs

    # ── 图片抽取 ─────────────────────────────────────────────────────
    def _extract_images(
        self,
        soup,
        image_map: dict[str, bytes],
        image_cache: dict[str, Optional[ExtractedImage]],
    ) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        seen: set[str] = set()
        for img_tag in soup.find_all("img"):
            src = (img_tag.get("src") or "").strip()
            if not src or src in seen:
                continue
            seen.add(src)
            img = self._resolve_image(src, image_map, image_cache)
            if img is not None:
                images.append(img)
        return images

    def _resolve_image(
        self,
        src: str,
        image_map: dict[str, bytes],
        image_cache: dict[str, Optional[ExtractedImage]],
    ) -> Optional[ExtractedImage]:
        if src in image_cache:
            return image_cache[src]

        result: Optional[ExtractedImage] = None
        for key in (src, os.path.basename(src), src.lstrip("./")):
            raw = image_map.get(key)
            if raw is None:
                continue
            result = self._prepare_image(raw)
            # 命中后同时按实际 key 缓存（同一文件的不同引用形式）
            image_cache[key] = result
            break

        image_cache[src] = result
        return result

    def _prepare_image(self, raw: bytes) -> Optional[ExtractedImage]:
        try:
            with Image.open(io.BytesIO(raw)) as img:
                width, height = img.size
                if width < self.MIN_IMAGE_SIZE or height < self.MIN_IMAGE_SIZE:
                    return None

                max_dim = settings.max_image_dimension
                if max(width, height) <= max_dim and img.format:
                    # 尺寸已达标：沿用原始编码，省掉一次 PNG 重编码
                    fmt = img.format.lower()
                    if fmt in ("png", "jpg", "jpeg"):
                        return ExtractedImage(data=raw, ext="jpg" if fmt == "jpeg" else fmt)
                    if fmt == "gif":
                        return ExtractedImage(data=raw, ext="gif")

                ratio = min(1.0, max_dim / max(width, height))
                resized = img.convert("RGB")
                if ratio < 1.0:
                    resized = resized.resize(
                        (max(1, int(width * ratio)), max(1, int(height * ratio))),
                        Image.LANCZOS,
                    )
                buf = io.BytesIO()
                resized.save(buf, format="PNG")
                return ExtractedImage(data=buf.getvalue(), ext="png")
        except Exception:
            return None

    @classmethod
    def supports(cls, _file_path: str) -> bool:
        return _file_path.lower().endswith(".epub")


_HEADING_RE = re.compile(r"^h[1-6]$")


def epub_read(file_path: str):
    from ebooklib import epub

    return epub.read_epub(file_path, options={"ignore_ncx": True})
