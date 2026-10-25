"""
摄入任务心跳。

摄入任务跑在进程内（asyncio.create_task），进程退出后台任务就没了，
文档会永久停在 pending / processing。这里用「磁盘心跳文件 + PID」判断任务是否还活着：

- 任务运行时每 N 秒写一次 {"pid": ..., "ts": ...}
- 启动清理时，PID 已不存在或心跳过期 → 视为僵尸任务

相比「一律标记为 failed」，这样在多 worker 场景下不会误杀别的 worker 正在跑的任务。
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from config import get_settings

settings = get_settings()

_HEARTBEAT_DIRNAME = "task_heartbeats"


def _heartbeat_dir() -> Path:
    path = Path(settings.storage_root) / _HEARTBEAT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _heartbeat_file(doc_id: str) -> Path:
    return _heartbeat_dir() / f"{doc_id}.json"


def beat(doc_id: str) -> None:
    """写入/刷新一次心跳（失败不影响主流程）"""
    try:
        _heartbeat_file(doc_id).write_text(
            json.dumps({"pid": os.getpid(), "ts": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        pass


def clear(doc_id: str) -> None:
    """任务结束时删除心跳文件"""
    try:
        _heartbeat_file(doc_id).unlink(missing_ok=True)
    except OSError:
        pass


def clear_all(doc_ids: list[str]) -> None:
    for doc_id in doc_ids:
        clear(doc_id)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但属于其他用户
        return True
    except OSError:
        return False
    return True


def _read(doc_id: str) -> dict:
    try:
        return json.loads(_heartbeat_file(doc_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def is_alive(doc_id: str, grace_seconds: Optional[int] = None) -> bool:
    """任务是否仍在运行：进程存在且心跳未过期"""
    grace = settings.stale_task_grace_seconds if grace_seconds is None else grace_seconds
    data = _read(doc_id)
    if not data:
        return False

    pid = data.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False

    if grace > 0:
        ts = float(data.get("ts") or 0)
        if time.time() - ts > grace:
            return False
    return True


def live_doc_ids(doc_ids: list[str], grace_seconds: Optional[int] = None) -> set[str]:
    """从给定的文档 ID 中筛出任务仍在运行的那些"""
    return {doc_id for doc_id in doc_ids if is_alive(doc_id, grace_seconds)}
