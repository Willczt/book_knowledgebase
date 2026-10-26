"""文档摄入队列：用数据库当队列，而不是裸 asyncio 任务。

裸 `asyncio.create_task` 的问题：进程重启任务就消失，文档永久卡在 processing；
多副本部署时没有任何互斥，同一文档可能被两个进程同时索引。

这里的最小可靠模型：
- claim：取一条 pending 且到期的文档，置为 processing、attempts+1、写租约
  （PostgreSQL 用 `FOR UPDATE SKIP LOCKED` 支持多副本并行消费）
- 续租：任务运行期间定期推后 `lease_until`，防止长任务被误判为死任务
- 恢复：租约过期的 processing 说明 worker 已死，重新入队
- 重试：未超过 `ingest_max_attempts` 时按指数退避设置 `next_retry_at`
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import structlog
from sqlalchemy import or_, select, update

from config import get_settings
from db.engine import AsyncSessionLocal
from db.models import Document, utcnow

logger = structlog.get_logger()
settings = get_settings()

_BACKOFF_BASE = 15
_BACKOFF_MAX = 300
_ERROR_MAX = 2000


async def claim_next() -> Optional[tuple[str, int]]:
    """认领一条待摄入文档，返回 (doc_id, attempts)。没有任务时返回 None。"""
    now = utcnow()
    lease = now + timedelta(seconds=settings.ingest_lease_seconds)

    async with AsyncSessionLocal() as db:
        async with db.begin():
            stmt = (
                select(Document.id)
                .where(
                    Document.status == "pending",
                    or_(Document.next_retry_at.is_(None), Document.next_retry_at <= now),
                )
                .order_by(Document.created_at)
                .limit(1)
            )
            if not settings.is_sqlite:
                stmt = stmt.with_for_update(skip_locked=True)

            doc_id = (await db.execute(stmt)).scalar_one_or_none()
            if doc_id is None:
                return None

            result = await db.execute(
                update(Document)
                .where(Document.id == doc_id, Document.status == "pending")
                .values(
                    status="processing",
                    attempts=Document.attempts + 1,
                    processing_started_at=now,
                    processing_completed_at=None,
                    lease_until=lease,
                    error_message=None,
                )
            )
            if result.rowcount == 0:
                return None

        attempts = (
            await db.execute(select(Document.attempts).where(Document.id == doc_id))
        ).scalar_one_or_none() or 1

    logger.info("ingest_claimed", doc_id=doc_id, attempt=attempts)
    return doc_id, attempts


async def renew_lease(doc_id: str) -> bool:
    """续租。返回 False 表示任务已不在 processing（可能被别的 worker 接管）。"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            update(Document)
            .where(Document.id == doc_id, Document.status == "processing")
            .values(
                lease_until=utcnow() + timedelta(seconds=settings.ingest_lease_seconds)
            )
        )
        await db.commit()
        return result.rowcount > 0


async def mark_done(doc_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(lease_until=None, next_retry_at=None)
        )
        await db.commit()


async def mark_failed(doc_id: str, attempts: int, error: str) -> None:
    """未超限则回到队列等待重试，超限则落 failed 终态。"""
    now = utcnow()
    exhausted = attempts >= settings.ingest_max_attempts

    if exhausted:
        values = {
            "status": "failed",
            "error_message": error[:_ERROR_MAX],
            "processing_completed_at": now,
            "lease_until": None,
            "next_retry_at": None,
        }
    else:
        backoff = min(_BACKOFF_MAX, _BACKOFF_BASE * 2 ** max(0, attempts - 1))
        values = {
            "status": "pending",
            "error_message": f"第 {attempts} 次尝试失败，{backoff}s 后重试：{error}"[:_ERROR_MAX],
            "next_retry_at": now + timedelta(seconds=backoff),
            "lease_until": None,
        }

    async with AsyncSessionLocal() as db:
        await db.execute(update(Document).where(Document.id == doc_id).values(**values))
        await db.commit()

    logger.warning(
        "ingest_task_failed",
        doc_id=doc_id,
        attempts=attempts,
        exhausted=exhausted,
        error=error[:300],
    )


async def recover_expired_leases() -> int:
    """把租约过期（worker 崩溃/被杀）的 processing 文档放回队列。"""
    now = utcnow()
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            update(Document)
            .where(
                Document.status == "processing",
                or_(Document.lease_until.is_(None), Document.lease_until < now),
            )
            .values(status="pending", lease_until=None, next_retry_at=now)
        )
        await db.commit()
        count = result.rowcount or 0

    if count:
        logger.warning("ingest_leases_recovered", count=count)
    return count


async def queue_depth() -> dict[str, int]:
    """供 /readyz 或运维使用：各状态文档数。"""
    from sqlalchemy import func

    async with AsyncSessionLocal() as db:
        rows = await db.execute(select(Document.status, func.count()).group_by(Document.status))
        return {status: count for status, count in rows.all()}
