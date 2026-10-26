"""rerank 服务（vLLM）的生命周期管理。

把 vLLM 当作本项目的子服务，统一约定：同一份配置（模型/端口/base_url）、
同一个 PID 文件、同一个日志文件。管理入口有三处，行为一致：

    python -m core.rerank_server status|up|down|restart|logs
    GET/POST /api/v1/rerank/...（前端或脚本可调用）
    RERANK_AUTOSTART=true（后端启动时自动拉起，退出时停止）

约定：
- 只有 `RERANK_BACKEND=http` 且 `RERANK_BASE_URL` 指向本机时才由本机管理；
  指向远程 GPU 机器时 start/stop 会明确拒绝，避免误杀。
- 端口从 `RERANK_BASE_URL` 解析，健康探针用 `http://127.0.0.1:<port>/health`。
- 子进程用 `start_new_session=True` 脱离当前进程组，后端重启不会连带杀掉 vLLM；
  但 `rerank_autostart` 开启时，shutdown 会主动停止它（reload 开发模式不建议开）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import structlog

try:  # Linux/WSL 才有，用于枚举本机网卡地址
    import fcntl
except ImportError:  # pragma: no cover - 非 POSIX 平台
    fcntl = None  # type: ignore[assignment]

from config import get_settings

logger = structlog.get_logger()

_PID_NAME = "rerank-server.pid"
_LOG_NAME = "rerank-server.log"
_STOP_TIMEOUT = 30.0
_HEALTH_TIMEOUT = 2.0
_POLL_INTERVAL = 2.0
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "::"}


class RerankServerError(RuntimeError):
    """rerank 服务无法按请求执行操作（配置不对 / 可执行文件缺失 / 启动失败）"""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _primary_ip() -> str:
    """本机默认路由出口地址（UDP connect 只查路由，不实际发包）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("223.5.5.5", 53))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _interface_ips() -> set[str]:
    """本机所有 IPv4 地址，用于判断 RERANK_BASE_URL 是否指向本机。"""
    ips: set[str] = set()
    if fcntl is not None:
        for _, name in socket.if_nameindex():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    packed = fcntl.ioctl(  # type: ignore[union-attr]
                        sock.fileno(), 0x8915, struct.pack("256s", name[:15].encode())
                    )
                    ips.add(socket.inet_ntoa(packed[20:24]))
            except OSError:
                continue
    if not ips:
        ips.add(_primary_ip())
    return ips


def _iter_proc_cmdlines():
    proc = Path("/proc")
    if not proc.is_dir():  # 非 Linux 环境不做进程发现
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        yield int(entry.name), [part.decode("utf-8", "replace") for part in raw.rstrip(b"\0").split(b"\0")]


class RerankServerManager:
    # ---------- 路径与端点 ----------

    @property
    def _storage_root(self) -> Path:
        """相对 storage_root 一律按项目根解析：CLI 从 backend/ 启动、后端从仓库根启动时落点一致。"""
        root = Path(get_settings().storage_root)
        if root.is_absolute():
            return root
        return Path(__file__).resolve().parents[2] / root

    @property
    def _run_dir(self) -> Path:
        return self._storage_root / "run"

    @property
    def _pid_path(self) -> Path:
        return self._run_dir / _PID_NAME

    @property
    def _log_path(self) -> Path:
        return self._storage_root / "logs" / _LOG_NAME

    def _endpoint(self) -> tuple[str, int]:
        """从 RERANK_BASE_URL 解析 (host, port)，端口即 vLLM 监听端口。"""
        parsed = urlparse(get_settings().rerank_base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8100
        return host, port

    @property
    def _health_url(self) -> str:
        host, port = self._endpoint()
        # 0.0.0.0 不能作为连接目标；WSL mirrored 网络下 loopback 也可能不通，用本机 IP 探针
        if host in {"0.0.0.0", "::"}:
            host = _primary_ip()
        return f"http://{host}:{port}/health"

    def is_local(self) -> bool:
        """base_url 指向本机（localhost / 本机网卡地址）时才由本机管理进程。"""
        host, _ = self._endpoint()
        if host in _LOCAL_HOSTS:
            return True
        return host in _interface_ips()

    def build_command(self) -> list[str]:
        s = get_settings()
        _, port = self._endpoint()
        executable = shutil.which(s.rerank_server_bin) or s.rerank_server_bin
        return [
            executable,
            "serve",
            s.rerank_model,
            "--host",
            s.rerank_server_bind,
            "--port",
            str(port),
            "--served-model-name",
            s.rerank_model,
            *shlex.split(s.rerank_server_extra_args or ""),
        ]

    # ---------- PID 文件 ----------

    def _read_pid(self) -> Optional[int]:
        try:
            data = json.loads(self._pid_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        pid = data.get("pid")
        return pid if isinstance(pid, int) else None

    def _write_pid(self, pid: int, command: list[str]) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._pid_path.write_text(
            json.dumps(
                {"pid": pid, "command": command, "started_at": datetime.now().isoformat()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _clear_pid(self) -> None:
        try:
            self._pid_path.unlink()
        except OSError:
            pass

    # ---------- 健康与进程发现 ----------

    async def _healthy(self, client: Optional[httpx.AsyncClient] = None) -> bool:
        try:
            if client is not None:
                resp = await client.get(self._health_url)
            else:
                async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as probe:
                    resp = await probe.get(self._health_url)
            return resp.status_code == 200
        except Exception:
            return False

    def _find_process(self) -> Optional[int]:
        """在 /proc 里找匹配当前模型 + 端口的 vllm serve 进程（手工启动的也能接管）。"""
        s = get_settings()
        _, port = self._endpoint()
        for pid, argv in _iter_proc_cmdlines():
            if not any("vllm" in part for part in argv):
                continue
            if "serve" not in argv or s.rerank_model not in argv:
                continue
            if f"--port={port}" not in argv:
                try:
                    index = argv.index("--port")
                except ValueError:
                    continue
                if index + 1 >= len(argv) or argv[index + 1] != str(port):
                    continue
            return pid
        return None

    async def status(self) -> dict[str, Any]:
        s = get_settings()
        pid = self._read_pid()
        if pid and not _pid_alive(pid):
            self._clear_pid()
            pid = None

        healthy = await self._healthy()
        adopted = False
        if pid is None and self.is_local():
            discovered = self._find_process()
            if discovered:
                pid, adopted = discovered, True

        return {
            "backend": s.rerank_backend,
            "enabled": s.rerank_enabled,
            "model": s.rerank_model,
            "base_url": s.rerank_base_url,
            "device_url": self._health_url,
            "local": self.is_local(),
            "running": bool(pid) or healthy,
            "healthy": healthy,
            "pid": pid,
            "adopted": adopted,  # True = 不是本管理器启动的（手工起的）
            "autostart": s.rerank_autostart,
            "log_path": str(self._log_path),
            "command": self.build_command() if self.is_local() else None,
        }

    # ---------- 启动 / 停止 ----------

    async def _wait_healthy(self, timeout: float, pid: Optional[int]) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self._healthy():
                return True
            if pid is not None and not _pid_alive(pid):
                raise RerankServerError(
                    "rerank 服务进程已退出，检查日志：" + str(self._log_path)
                )
            await asyncio.sleep(_POLL_INTERVAL)
        return False

    async def start(self, wait: bool = True) -> dict[str, Any]:
        s = get_settings()
        if (s.rerank_backend or "").lower() != "http":
            raise RerankServerError(
                f"当前 RERANK_BACKEND={s.rerank_backend}，无需启动 vLLM 服务"
            )
        if not self.is_local():
            raise RerankServerError(
                f"RERANK_BASE_URL={s.rerank_base_url} 指向远程服务，本机不负责启动"
            )

        if await self._healthy():
            return {**await self.status(), "message": "已经在运行"}

        pid = self._read_pid()
        if pid and _pid_alive(pid):
            ok = await self._wait_healthy(s.rerank_start_timeout, pid) if wait else False
            return {
                **await self.status(),
                "message": "启动中，已就绪" if ok else "启动中",
            }

        command = self.build_command()
        if not (Path(command[0]).is_file() or shutil.which(command[0])):
            raise RerankServerError(
                f"找不到 vLLM 可执行文件：{command[0]}，请设置 RERANK_SERVER_BIN 为绝对路径"
            )

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[2]
        with self._log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # 独立会话：后端重启不会带走它
                cwd=str(project_root),
            )
        self._write_pid(process.pid, command)
        logger.info(
            "rerank_server_started",
            pid=process.pid,
            port=self._endpoint()[1],
            model=s.rerank_model,
            log=str(self._log_path),
        )

        ready = await self._wait_healthy(s.rerank_start_timeout, process.pid) if wait else False
        return {
            **await self.status(),
            "message": "已启动并健康" if ready else "已启动，仍在加载模型",
        }

    def _terminate(self, pid: int) -> None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _kill(self, pid: int) -> None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def _wait_dead(self, pid: int, timeout: float = _STOP_TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return True
            await asyncio.sleep(0.5)
        return False

    async def stop(self) -> dict[str, Any]:
        if not self.is_local():
            raise RerankServerError(
                f"RERANK_BASE_URL={get_settings().rerank_base_url} 指向远程服务，本机不负责停止"
            )

        pid = self._read_pid()
        adopted = False
        if pid is None or not _pid_alive(pid):
            discovered = self._find_process()
            pid, adopted = discovered, discovered is not None

        if pid is None:
            return {**await self.status(), "message": "没有发现运行中的 rerank 服务"}

        self._terminate(pid)
        died = await self._wait_dead(pid)
        if not died:
            self._kill(pid)
            died = await self._wait_dead(pid, 5)
        self._clear_pid()
        logger.info("rerank_server_stopped", pid=pid, forced=not died, adopted=adopted)
        if not died:
            message = "已发送终止信号，进程可能仍在退出"
        elif adopted:
            message = "已停止（该实例不是本管理器启动的，已一并接管停止）"
        else:
            message = "已停止"
        return {**await self.status(), "message": message}

    async def restart(self, wait: bool = True) -> dict[str, Any]:
        await self.stop()
        return await self.start(wait=wait)

    # ---------- 日志 ----------

    def tail_log(self, lines: int = 80) -> list[str]:
        if not self._log_path.is_file():
            return []
        with self._log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            data = fh.read().decode("utf-8", "replace")
        return data.splitlines()[-lines:]


_manager: Optional[RerankServerManager] = None


def get_server() -> RerankServerManager:
    global _manager
    if _manager is None:
        _manager = RerankServerManager()
    return _manager


def reset_server() -> None:
    """测试用：丢弃单例（不停止进程）"""
    global _manager
    _manager = None


# ========================
# 命令行入口：python -m core.rerank_server <command>
# ========================

async def _amain(args: argparse.Namespace) -> int:
    server = get_server()
    if args.command == "status":
        print(json.dumps(await server.status(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "up":
        print(json.dumps(await server.start(wait=not args.no_wait), ensure_ascii=False, indent=2))
        return 0
    if args.command == "down":
        print(json.dumps(await server.stop(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "restart":
        print(json.dumps(await server.restart(wait=not args.no_wait), ensure_ascii=False, indent=2))
        return 0
    if args.command == "logs":
        for line in server.tail_log(args.lines):
            print(line)
        return 0
    raise SystemExit(f"未知命令：{args.command}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m core.rerank_server", description="管理 vLLM rerank 服务")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="查看运行状态与健康检查")
    up = sub.add_parser("up", help="启动服务（默认等待就绪）")
    up.add_argument("--no-wait", action="store_true", help="启动后立即返回，不等待健康检查")
    sub.add_parser("down", help="停止服务")
    restart = sub.add_parser("restart", help="重启服务")
    restart.add_argument("--no-wait", action="store_true", help="重启后立即返回")
    logs = sub.add_parser("logs", help="查看服务日志尾部")
    logs.add_argument("-n", "--lines", type=int, default=80)
    args = parser.parse_args()

    try:
        return asyncio.run(_amain(args))
    except RerankServerError as exc:
        print(f"错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
