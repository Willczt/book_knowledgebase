import os
import re
from typing import Iterator

from core.parsers.base import DocumentParser, ParsedChunk


class TXTParser(DocumentParser):
    """
    解析纯文本（网络小说为主）：
    - 自动识别编码（UTF-8 / UTF-16 BOM / GB18030 / Big5）
    - 按「第N章 / 序 / 楔子 / 番外」等标题切分章节
    - 清理全角空格缩进与多余空行
    """

    # 单段最大字符数：文件无章节标记时，按此长度切片，避免单块过大
    MAX_SECTION_CHARS = 20000
    # 章节标题行最大长度，避免把正文里提到"第一章"的句子当标题
    MAX_TITLE_LENGTH = 40

    _CHAPTER_RE = re.compile(
        r"^(?:"
        r"第\s*[0-9０-９零〇一二三四五六七八九十百千万两]+\s*[章节回卷集部篇话]"
        r"|楔子|序章|序言|序|前言|引子|后记|尾声|终章|番外|外传|卷[0-9０-９零一二三四五六七八九十百千万两]+"
        r")"
    )
    _TRAILING_PUNCT = "。，！？；：、…—,.!?;:"

    def parse(self, file_path: str) -> Iterator[ParsedChunk]:
        filename = os.path.basename(file_path)
        content = self._read_text(file_path)

        for title, text in self._split_chapters(content):
            if not text.strip():
                continue
            yield ParsedChunk(
                text=text,
                chapter=title,
                source_doc=filename,
            )

    # ── 编码与文本清理 ───────────────────────────────────────────────
    @staticmethod
    def _read_text(file_path: str) -> str:
        with open(file_path, "rb") as f:
            raw = f.read()

        # BOM 优先
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig")
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")

        for encoding in ("utf-8", "gb18030", "big5"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    @classmethod
    def _normalize(cls, content: str) -> list[str]:
        """按行清理：去缩进、去空行、去重复空行，返回非空行列表"""
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for raw_line in content.split("\n"):
            line = raw_line.replace("\u3000", " ").strip()
            if line:
                lines.append(line)
        return lines

    @classmethod
    def _is_chapter_title(cls, line: str) -> bool:
        if not line or len(line) > cls.MAX_TITLE_LENGTH:
            return False
        if line[-1] in cls._TRAILING_PUNCT:
            return False
        return bool(cls._CHAPTER_RE.match(line))

    # ── 章节切分 ─────────────────────────────────────────────────────
    def _split_chapters(self, content: str) -> list[tuple[str, str]]:
        lines = self._normalize(content)

        sections: list[tuple[str, list[str]]] = []
        current_title: str = ""
        current_lines: list[str] = []

        for line in lines:
            if self._is_chapter_title(line):
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = line
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, current_lines))

        if not sections:
            return []

        # 没有任何章节标题 → 按长度切片，章节名留空
        if len(sections) == 1 and not sections[0][0]:
            return [(title, text) for title, text in self._chunk_by_size(sections[0][1])]

        return [
            (title, "\n\n".join(chapter_lines))
            for title, chapter_lines in sections
        ]

    def _chunk_by_size(self, lines: list[str]) -> list[tuple[str, str]]:
        blocks: list[tuple[str, str]] = []
        buffer: list[str] = []
        size = 0

        for line in lines:
            buffer.append(line)
            size += len(line) + 1
            if size >= self.MAX_SECTION_CHARS:
                blocks.append(("", "\n\n".join(buffer)))
                buffer = []
                size = 0

        if buffer:
            blocks.append(("", "\n\n".join(buffer)))
        return blocks

    @classmethod
    def supports(cls, _file_path: str) -> bool:
        return _file_path.lower().endswith((".txt", ".text"))
