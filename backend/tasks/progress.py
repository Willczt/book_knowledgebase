"""
摄入进度（进程内存，供前端轮询展示）。

只存当前进程内正在跑的任务；进程重启后由启动清理把文档标记为 failed，
所以这里不需要持久化。
"""

import time
from typing import Optional

_PROGRESS: dict[str, dict] = {}


def start(doc_id: str) -> None:
    _PROGRESS[doc_id] = {
        "stage": "queued",
        "stage_label": "排队中",
        "percent": 0.0,
        "current": None,
        "total": None,
        "message": None,
        "updated_at": time.time(),
    }


def update(doc_id: str, payload: dict) -> None:
    """合并一次进度更新（payload 来自 Indexer 的回调）"""
    entry = _PROGRESS.setdefault(doc_id, {})
    entry.update(payload)
    entry["updated_at"] = time.time()


def get(doc_id: str) -> Optional[dict]:
    return _PROGRESS.get(doc_id)


def finish(doc_id: str) -> None:
    _PROGRESS.pop(doc_id, None)
