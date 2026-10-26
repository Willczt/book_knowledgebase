"""BM25 稀疏向量编码器（进程内单例）。

FastEmbed 的 SparseTextEmbedding 每次构造都会加载一次 ONNX 模型（数百 MB + 秒级耗时），
摄入和每次检索都新建实例会让内存和首个 token 延迟都失控，所以这里做进程级缓存。
"""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_sparse_encoder() -> Any:
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_sparse(
    texts: list[str],
    batch_size: int = 32,
) -> list[tuple[list[int], list[float]]]:
    """同步编码，调用方应放到线程池（asyncio.to_thread）执行。

    返回与输入顺序一致的 [(indices, values), ...]。
    """
    if not texts:
        return []

    encoder = get_sparse_encoder()
    results: list[tuple[list[int], list[float]]] = []
    for emb in encoder.embed(texts, batch_size=batch_size):
        results.append((emb.indices.tolist(), emb.values.tolist()))
    return results
