"""知识库导出的后台任务与进度状态。

导出要遍历向量库和全部文档，大库可能几十秒到几分钟。同步接口下浏览器拿不到
任何反馈，用户只会看到按钮点了没反应。这里改成「后台任务 + 轮询进度」：
接口立即返回 job_id，前端每 600ms 拉一次百分比和阶段文案，完成后走下载接口。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 任务与临时包保留时间：过期后自动清理，避免磁盘堆积
JOB_TTL_SECONDS = 30 * 60

_JOBS: dict[str, "ExportJob"] = {}


@dataclass
class ExportJob:
    job_id: str
    kb_id: str
    kb_name: str
    filename: str
    status: str = "running"  # running | done | failed
    percent: float = 0.0
    stage: str = "准备中"
    path: Optional[Path] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    downloaded: bool = False

    @property
    def elapsed_s(self) -> float:
        return round(time.time() - self.created_at, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kb_id": self.kb_id,
            "kb_name": self.kb_name,
            "status": self.status,
            "percent": round(self.percent, 1),
            "stage": self.stage,
            "filename": self.filename,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "download_ready": self.status == "done" and self.path is not None,
            "downloaded": self.downloaded,
        }


def _discard_file(job: ExportJob) -> None:
    path = job.path
    job.path = None
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def cleanup() -> None:
    """清理过期任务与临时文件（顺手删掉已下载的包）。"""
    now = time.time()
    for job_id, job in list(_JOBS.items()):
        expired = now - job.created_at > JOB_TTL_SECONDS
        if job.downloaded or expired:
            _discard_file(job)
        if expired:
            _JOBS.pop(job_id, None)


def create_job(kb_id: str, kb_name: str, filename: str) -> ExportJob:
    cleanup()
    job = ExportJob(
        job_id=uuid.uuid4().hex[:16],
        kb_id=kb_id,
        kb_name=kb_name,
        filename=filename,
    )
    _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[ExportJob]:
    return _JOBS.get(job_id)


def find_running(kb_id: str) -> Optional[ExportJob]:
    """同一知识库同时只跑一个导出任务。"""
    for job in _JOBS.values():
        if job.kb_id == kb_id and job.status == "running":
            return job
    return None


def update(job: ExportJob, *, percent: Optional[float] = None,
           stage: Optional[str] = None) -> None:
    if percent is not None:
        job.percent = max(0.0, min(100.0, percent))
    if stage is not None:
        job.stage = stage
    job.updated_at = time.time()


def finish(job: ExportJob, path: Path) -> None:
    job.path = path
    job.status = "done"
    job.percent = 100.0
    job.stage = "打包完成，开始下载"
    job.updated_at = time.time()


def fail(job: ExportJob, error: str) -> None:
    _discard_file(job)
    job.status = "failed"
    job.error = error[:300]
    job.stage = "导出失败"
    job.updated_at = time.time()


def mark_downloaded(job: ExportJob) -> None:
    job.downloaded = True
    _discard_file(job)
