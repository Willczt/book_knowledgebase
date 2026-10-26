"""摄入队列：认领、重试退避、租约恢复与 worker 失败处理。"""

from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.models import Base, Document, utcnow
from tasks import queue
from tasks import worker as worker_mod


@pytest.fixture
async def factory(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/queue.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(queue, "AsyncSessionLocal", session_factory)
    yield session_factory
    await engine.dispose()


async def _add_doc(factory, doc_id: str, **kwargs) -> None:
    async with factory() as db:
        db.add(
            Document(
                id=doc_id,
                kb_id="kb-test",
                filename=f"{doc_id}.txt",
                file_type="txt",
                file_path=f"/tmp/{doc_id}.txt",
                status=kwargs.pop("status", "pending"),
                **kwargs,
            )
        )
        await db.commit()


async def _get(factory, doc_id: str) -> Document:
    async with factory() as db:
        return (
            await db.execute(select(Document).where(Document.id == doc_id))
        ).scalar_one()


async def test_claim_marks_processing_and_counts_attempt(factory):
    await _add_doc(factory, "d1")

    claimed = await queue.claim_next()
    assert claimed == ("d1", 1)

    doc = await _get(factory, "d1")
    assert doc.status == "processing"
    assert doc.attempts == 1
    assert doc.lease_until is not None
    assert doc.lease_until > utcnow()

    # 已被认领，第二次拿不到
    assert await queue.claim_next() is None


async def test_failed_attempt_goes_back_with_backoff(factory):
    await _add_doc(factory, "d2")
    doc_id, attempts = await queue.claim_next()

    await queue.mark_failed(doc_id, attempts, "boom")
    doc = await _get(factory, "d2")
    assert doc.status == "pending"
    assert doc.attempts == 1
    assert doc.next_retry_at is not None and doc.next_retry_at > utcnow()
    assert "boom" in (doc.error_message or "")

    # 退避期内不应该被重新认领
    assert await queue.claim_next() is None


async def test_retry_stops_after_max_attempts(factory, monkeypatch):
    monkeypatch.setattr(queue.settings, "ingest_max_attempts", 2)
    await _add_doc(factory, "d3")

    first = await queue.claim_next()
    await queue.mark_failed(*first, "first failure")

    # 模拟退避到期
    async with factory() as db:
        doc = (
            await db.execute(select(Document).where(Document.id == "d3"))
        ).scalar_one()
        doc.next_retry_at = utcnow() - timedelta(seconds=1)
        await db.commit()

    second = await queue.claim_next()
    assert second == ("d3", 2)
    await queue.mark_failed(*second, "second failure")

    doc = await _get(factory, "d3")
    assert doc.status == "failed"
    assert doc.lease_until is None
    assert "second failure" in (doc.error_message or "")


async def test_recover_expired_lease_requeues_document(factory):
    await _add_doc(
        factory,
        "d4",
        status="processing",
        attempts=1,
        lease_until=utcnow() - timedelta(minutes=5),
    )

    recovered = await queue.recover_expired_leases()
    assert recovered == 1

    doc = await _get(factory, "d4")
    assert doc.status == "pending"
    assert doc.lease_until is None

    assert (await queue.claim_next())[0] == "d4"


async def test_renew_lease_extends_deadline(factory):
    await _add_doc(factory, "d5")
    doc_id, _ = await queue.claim_next()
    before = (await _get(factory, "d5")).lease_until

    assert await queue.renew_lease(doc_id) is True
    after = (await _get(factory, "d5")).lease_until
    assert after > before

    # 任务已结束（改成 indexed）后不应再续租
    async with factory() as db:
        doc = (
            await db.execute(select(Document).where(Document.id == "d5"))
        ).scalar_one()
        doc.status = "indexed"
        await db.commit()
    assert await queue.renew_lease(doc_id) is False


async def test_worker_marks_retry_on_ingest_failure(factory, monkeypatch):
    await _add_doc(factory, "d6")

    async def _boom(doc_id: str) -> None:
        raise RuntimeError("parse failed")

    monkeypatch.setattr(worker_mod, "run_ingest_task", _boom)

    ingest_worker = worker_mod.IngestWorker(concurrency=1)
    claimed = await queue.claim_next()
    await ingest_worker._process(*claimed)

    doc = await _get(factory, "d6")
    assert doc.status == "pending"
    assert "parse failed" in (doc.error_message or "")
    assert doc.next_retry_at is not None
