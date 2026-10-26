"""中文友好的 BM25 稀疏向量编码。

FastEmbed 的 ``Qdrant/bm25`` 用 ``\\w+`` 切词，中文整句会被当成一个 token，
索引与查询各自都只有一个「长词」，因此中文 BM25 基本 0 命中，混合检索退化
成纯向量检索。

这里改成自实现的编码器，规则固定、进程间一致：
- NFKC 归一化 + 小写；
- 英文/数字按连续串切词（长度 >= 2，纯数字保留）；
- 中文按「单字 + 相邻二元组」切分（不引入 jieba 等额外分词依赖）；
- 每个 term 用 mmh3 稳定哈希成 uint32 作为稀疏向量下标；
- 值取 BM25 的词频饱和项 ``tf*(k1+1)/(tf + k1*(1-b+b*len/avg_len))``，
  IDF 交给 Qdrant 稀疏索引的 ``modifier=idf`` 在服务端计算。

注意：编码规则变了以后，旧数据必须用 ``scripts/rebuild_sparse.py`` 重建稀疏向量，
否则查询（新分词）和库里（旧分词）对不上。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import mmh3

# 编码规则版本：变更分词/权重时需要重建索引
TOKENIZER_VERSION = 2

K1 = 1.2
B = 0.75
AVG_LEN = 256.0
# 单个 chunk 保留的 term 上限：超出部分权重极低，截断可控制向量体积
MAX_TERMS = 512

_CJK_RANGES = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_TOKEN_RE = re.compile(rf"[{_CJK_RANGES}]+|[a-z0-9]+")
_CJK_RE = re.compile(rf"[{_CJK_RANGES}]")

# 高频虚词：单字层面过滤，二元组保留（"的关" 这类组合仍可能有区分度）
_CJK_STOPWORDS = set("的了是在和与有我你他她它这那就也都而及或但被把对从到为以等之其于与")
_EN_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "with", "as", "at", "by",
}


def tokenize(text: str) -> list[str]:
    """把文本切成可哈希的 term 序列（英文词 + 中文单字/二元组）。"""
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()

    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        piece = match.group(0)
        if _CJK_RE.match(piece):
            chars = list(piece)
            tokens.extend(char for char in chars if char not in _CJK_STOPWORDS)
            tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        else:
            if piece in _EN_STOPWORDS:
                continue
            if len(piece) >= 2 or piece.isdigit():
                tokens.append(piece)
    return tokens


def encode(text: str) -> tuple[list[int], list[float]]:
    """把一段文本编码为 (indices, values)，indices 升序，便于比较与存储。"""
    tokens = tokenize(text)
    if not tokens:
        return [], []

    counts = Counter(tokens)
    doc_len = len(tokens)
    length_norm = K1 * (1 - B + B * doc_len / AVG_LEN)

    merged: dict[int, float] = {}
    for token, tf in counts.items():
        weight = tf * (K1 + 1) / (tf + length_norm)
        index = mmh3.hash(token, signed=False)
        merged[index] = merged.get(index, 0.0) + weight

    top = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:MAX_TERMS]
    top.sort(key=lambda item: item[0])
    return [index for index, _ in top], [round(weight, 6) for _, weight in top]


def embed_sparse(
    texts: list[str],
    batch_size: int = 32,
) -> list[tuple[list[int], list[float]]]:
    """同步编码，调用方应放到线程池（asyncio.to_thread）执行。

    返回与输入顺序一致的 [(indices, values), ...]；batch_size 仅为兼容旧调用保留。
    """
    if not texts:
        return []
    return [encode(text) for text in texts]
