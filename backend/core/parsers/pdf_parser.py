import io
import os
from typing import Iterator

from PIL import Image

from config import get_settings
from core.parsers.base import DocumentParser, ExtractedImage, ParsedChunk

settings = get_settings()


class PDFParser(DocumentParser):
    """使用 PyMuPDF (fitz) 解析非扫描 PDF：提取文字和嵌入图片"""

    MIN_IMAGE_SIZE = 50     # 忽略小于 50x50 的图片（图标、装饰）
    MIN_TEXT_LENGTH = 20    # 忽略少于 20 字符的段落

    def parse(self, file_path: str) -> Iterator[ParsedChunk]:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)

        # 提取文档元数据
        metadata = doc.metadata or {}

        for page_num, page in enumerate(doc, start=1):
            # 提取页面文字（保留段落结构）
            blocks = page.get_text("blocks")  # [(x0, y0, x1, y1, text, block_no, type)]
            page_text_parts = []
            for block in blocks:
                if block[6] == 0:  # type 0 = 文字块
                    text = block[4].strip()
                    if len(text) >= self.MIN_TEXT_LENGTH:
                        page_text_parts.append(text)

            page_text = "\n\n".join(page_text_parts)

            # 提取页面图片
            images: list[ExtractedImage] = []
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_ext = base_image.get("ext", "png")

                    # 检查图片尺寸
                    img = Image.open(io.BytesIO(img_bytes))
                    w, h = img.size
                    if w < self.MIN_IMAGE_SIZE or h < self.MIN_IMAGE_SIZE:
                        continue

                    # 统一转换为 PNG
                    if img_ext.lower() not in ("png", "jpg", "jpeg"):
                        buf = io.BytesIO()
                        img.convert("RGB").save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                        img_ext = "png"

                    # 调整最大尺寸以降低 LLM caption 成本
                    max_dim = settings.max_image_dimension
                    if max(w, h) > max_dim:
                        ratio = max_dim / max(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        buf = io.BytesIO()
                        img.resize(new_size, Image.LANCZOS).save(buf, format="PNG")
                        img_bytes = buf.getvalue()

                    images.append(ExtractedImage(
                        data=img_bytes,
                        ext=img_ext,
                        page_num=page_num,
                    ))
                except Exception:
                    continue

            if page_text or images:
                yield ParsedChunk(
                    text=page_text,
                    images=images,
                    page_num=page_num,
                    source_doc=filename,
                    extra_metadata={
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "total_pages": doc.page_count,
                    },
                )

        doc.close()

    @classmethod
    def supports(cls, _file_path: str) -> bool:
        return _file_path.lower().endswith(".pdf")
