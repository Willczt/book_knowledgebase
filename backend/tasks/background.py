"""后台任务注册表。

`asyncio.create_task()` 返回的任务如果没有任何强引用，可能在执行过程中被 GC 回收，
导致摄入中断或消息静默丢失。这里统一持有引用，任务结束自动释放。
"""

import asyncio
from typing import Any, Coroutine

_tasks: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """创建并持有后台任务，异常交给任务自身处理/记录日志。"""
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_on_done)
    return task


def _on_done(task: asyncio.Task) -> None:
    _tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        # done_callback 不在 except 块中，exc_info 拿不到异常，需要手动格式化堆栈
        import traceback

        import structlog

        structlog.get_logger().error(
            "background_task_failed",
            error=f"{type(exc).__name__}: {exc}"[:500],
            traceback="".join(traceback.format_exception(exc))[:2000],
        )
