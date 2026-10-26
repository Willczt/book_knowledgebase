from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import get_settings

settings = get_settings()

# 确保 SQLite 存储目录存在
if settings.is_sqlite:
    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

_engine_options: dict = {
    "echo": False,
    "future": True,
    # 生产必须开启：数据库重启/网络抖动后自动剔除坏连接
    "pool_pre_ping": True,
}

if settings.is_sqlite:
    # SQLite：单进程内共享连接，aiosqlite 不支持多线程池参数
    _engine_options["connect_args"] = {"check_same_thread": False}
else:
    _engine_options.update(
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_timeout=30,
    )

engine = create_async_engine(settings.database_url, **_engine_options)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# 线上正常应由 Alembic 管表结构；这里只做「create_all + 缺失列补齐」，
# 保证从旧版本升级时不用手工 ALTER（新增列必须可空或有默认值）。
_ADDED_COLUMNS: dict[str, str] = {
    "attempts": "INTEGER NOT NULL DEFAULT 0",
    "next_retry_at": "TIMESTAMP",
    "lease_until": "TIMESTAMP",
}


def _sync_schema(sync_conn) -> None:
    inspector = inspect(sync_conn)
    if "documents" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("documents")}
    for name, ddl in _ADDED_COLUMNS.items():
        if name not in existing:
            sync_conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {ddl}"))

    # 队列扫描索引（SQLite / PostgreSQL 都支持 IF NOT EXISTS）
    sync_conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_documents_queue "
            "ON documents (status, next_retry_at)"
        )
    )


async def init_db() -> None:
    from db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_sync_schema)


async def dispose_db() -> None:
    await engine.dispose()


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
