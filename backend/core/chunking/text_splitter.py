from dataclasses import dataclass
from typing import Optional


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_num: Optional[int] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None


class RecursiveTextSplitter:
    """
    递归字符文本分割器。
    优先在段落边界分割，回退到句子边界，最后按字符。
    chunk_size 以字符数计（适合中英混排）。
    """

    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(
        self,
        text: str,
        page_num: Optional[int] = None,
        section_title: Optional[str] = None,
        chapter: Optional[str] = None,
    ) -> list[TextChunk]:
        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    chunk_index=i,
                    page_num=page_num,
                    section_title=section_title,
                    chapter=chapter,
                ))
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        separator = separators[-1]
        for sep in separators:
            if sep and sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)

        # 合并小块
        merged: list[str] = []
        current = ""
        for split in splits:
            candidate = (current + separator + split).strip() if current else split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # 太长的块继续递归分割
                if len(split) > self.chunk_size:
                    remaining_seps = separators[separators.index(separator) + 1:] if separator in separators else separators[1:]
                    if remaining_seps:
                        merged.extend(self._split_recursive(split, remaining_seps))
                    else:
                        # 强制按字符切分
                        for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                            merged.append(split[i: i + self.chunk_size])
                    current = ""
                else:
                    current = split

        if current:
            merged.append(current)

        # 添加 overlap
        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped = [merged[0]]
            for i in range(1, len(merged)):
                prev = merged[i - 1]
                overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
                overlapped.append(overlap_text + " " + merged[i])
            return overlapped

        return merged
