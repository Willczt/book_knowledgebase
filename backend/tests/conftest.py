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
    """只实现导入导出用到的方法"""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict]] = {}

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
