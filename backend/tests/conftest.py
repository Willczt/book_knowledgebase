"""测试公共 fixture：临时 SQLite + 假的 Qdrant。

不依赖真实 Qdrant / 外部模型服务，保证 CI 可离线运行。
"""

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# 让 tests 能直接 import 后端包
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core import transfer  # noqa: E402
from db.models import Base  # noqa: E402


class FakeQdrant:
    """只实现导入导出 / 检索用到的少数方法"""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict]] = {}
        # 检索测试用：collection -> 预置的 SearchResult 列表
        self.results: dict[str, list] = {}
        # 最近一次 hybrid_search 的入参，便于断言候选池大小
        self.last_search: dict = {}

    async def create_collection(self, name: str, dim: int) -> None:
        self.collections.setdefault(name, [])

    async def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    async def iter_points(self, name: str, batch_size: int = 256):
        for point in list(self.collections.get(name, [])):
            yield point

    async def scroll_all(self, name: str, batch_size: int = 256) -> list[dict]:
        return [p async for p in self.iter_points(name, batch_size)]

    async def upsert_raw(self, name: str, points: list[dict]) -> int:
        self.collections.setdefault(name, []).extend(points)
        return len(points)

    async def count(self, name: str) -> int:
        return len(self.collections.get(name, []))

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 10,
        prefetch_limit: int = 20,
        doc_ids_filter: list[str] | None = None,
        debug_sink: dict | None = None,
    ) -> list:
        self.last_search = {
            "collection": collection_name,
            "top_k": top_k,
            "prefetch_limit": prefetch_limit,
            "doc_ids_filter": doc_ids_filter,
        }
        hits = list(self.results.get(collection_name, []))[:top_k]
        if debug_sink is not None:
            # 调试采集：按结果顺序伪造两路分数与原始排名
            debug_sink["fusion"] = {"method": "rrf", "k": 60, "prefetch_limit": prefetch_limit}
            debug_sink["vector"] = {
                hit.chunk_id: {"score": 0.9 - index * 0.05, "rank": index + 1}
                for index, hit in enumerate(hits)
            }
            debug_sink["sparse"] = {
                hit.chunk_id: {"score": 0.5 - index * 0.05, "rank": index + 1}
                for index, hit in enumerate(hits)
            }
        return hits


@pytest.fixture
def qdrant() -> FakeQdrant:
    return FakeQdrant()


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def storage(tmp_path, monkeypatch) -> Path:
    """把 storage_root 指到临时目录，避免污染真实数据"""
    monkeypatch.setattr(transfer.settings, "storage_root", str(tmp_path))
    return tmp_path
