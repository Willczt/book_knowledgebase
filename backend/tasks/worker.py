"""进程内摄入 worker：从 DB 队列取任务并执行。

- 并发固定为 `INGEST_WORKERS`，避免一次上传几十个文件把 embedding 服务打满
- 任务执行期间定期续租，进程被杀后租约过期，文档会自动回到队列
- 优雅关闭：先停止认领，给在跑任务 `SHUTDOWN_GRACE_SECONDS` 时间收尾
"""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

from config import get_settings
from tasks import queue
from tasks.ingest_task import run_ingest_task

logger = structlog.get_logger()
settings = get_settings()


class IngestWorker:
    def __init__(self, concurrency: Optional[int] = None) -> None:
        self.concurrency = max(1, concurrency or settings.ingest_workers)
        self._tasks: list[asyncio.Task] = []
        self._wakeup = asyncio.Event()
        self._stopping = False

    async def start(self) -> None:
        await queue.recover_expired_leases()
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._loop(index), name=f"ingest-worker-{index}")
            for index in range(self.concurrency)
        ]
        logger.info("ingest_worker_started", workers=self.concurrency)

    def notify(self) -> None:
        """有新的待办任务时唤醒空闲 worker（不需要等轮询）。"""
        self._wakeup.set()

    async def stop(self) -> None:
        self._stopping = True
        self._wakeup.set()
        if not self._tasks:
            return

        logger.info("ingest_worker_stopping", workers=len(self._tasks))
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=settings.shutdown_grace_seconds,
            )
        except TimeoutError:
            logger.warning(
                "ingest_worker_shutdown_timeout",
                timeout=settings.shutdown_grace_seconds,
            )
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            self._tasks.clear()

    async def _loop(self, index: int) -> None:
        while not self._stopping:
            try:
                claimed = await queue.claim_next()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "ingest_claim_failed",
                    worker=index,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
                await self._sleep(5.0)
                continue

            if claimed is None:
                await self._wait_for_work()
                continue

            doc_id, attempts = claimed
            await self._process(doc_id, attempts)

    async def _process(self, doc_id: str, attempts: int) -> None:
        stop_renew = asyncio.Event()
        renew_task = asyncio.create_task(self._renew_loop(doc_id, stop_renew))
        try:
            await run_ingest_task(doc_id)
        except asyncio.CancelledError:
            # 关闭超时被强制取消：保留租约，下次启动由恢复逻辑重新入队
            raise
        except Exception as exc:
            await queue.mark_failed(doc_id, attempts, f"{type(exc).__name__}: {exc}")
        else:
            await queue.mark_done(doc_id)
            logger.info("ingest_task_finished", doc_id=doc_id, attempts=attempts)
        finally:
            stop_renew.set()
            renew_task.cancel()
            await asyncio.gather(renew_task, return_exceptions=True)

    async def _renew_loop(self, doc_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.ingest_lease_renew_seconds)
                return
            except TimeoutError:
                try:
                    alive = await queue.renew_lease(doc_id)
                except Exception as exc:
                    logger.warning(
                        "ingest_lease_renew_failed",
                        doc_id=doc_id,
                        error=str(exc)[:200],
                    )
                    continue
                if not alive:
                    logger.warning("ingest_lease_lost", doc_id=doc_id)
                    return

    async def _wait_for_work(self) -> None:
        self._wakeup.clear()
        try:
            await asyncio.wait_for(self._wakeup.wait(), timeout=settings.ingest_poll_seconds)
        except TimeoutError:
            # 轮询兜底：处理 next_retry_at 到期，或多副本下别人写入的任务
            return

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._wakeup.wait(), timeout=seconds)
        except TimeoutError:
            return


_worker: Optional[IngestWorker] = None


def get_worker() -> IngestWorker:
    global _worker
    if _worker is None:
        _worker = IngestWorker()
    return _worker
