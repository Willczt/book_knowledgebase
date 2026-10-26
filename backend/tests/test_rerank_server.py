"""rerank 服务（vLLM）生命周期管理：命令构建、状态、启停与进程发现。"""

import json
import os
import subprocess
import sys

import pytest

from core import rerank_server
from core.rerank_server import RerankServerError


@pytest.fixture
def server(tmp_path, monkeypatch):
    settings = rerank_server.get_settings()
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))
    monkeypatch.setattr(settings, "rerank_backend", "http")
    monkeypatch.setattr(settings, "rerank_base_url", "http://localhost:8100/v1")
    monkeypatch.setattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setattr(settings, "rerank_server_bin", sys.executable)
    monkeypatch.setattr(settings, "rerank_server_bind", "0.0.0.0")
    monkeypatch.setattr(
        settings,
        "rerank_server_extra_args",
        "--gpu-memory-utilization 0.3 --max-model-len 2048",
    )
    monkeypatch.setattr(settings, "rerank_start_timeout", 5.0)
    monkeypatch.setattr(rerank_server, "_POLL_INTERVAL", 0.01)
    rerank_server.reset_server()
    return rerank_server.get_server()


def _patch_health(monkeypatch, instance, values):
    """按顺序返回健康检查结果（用完后保持最后一次）。"""
    queue = list(values)

    async def fake_healthy(client=None):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(instance, "_healthy", fake_healthy)


def test_build_command(server):
    cmd = server.build_command()
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["serve", "BAAI/bge-reranker-v2-m3"]
    assert cmd[cmd.index("--port") + 1] == "8100"
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert "--gpu-memory-utilization" in cmd and "0.3" in cmd
    assert "--max-model-len" in cmd and "2048" in cmd


async def test_status_without_service(server, monkeypatch):
    _patch_health(monkeypatch, server, [False])
    monkeypatch.setattr(server, "_find_process", lambda: None)

    state = await server.status()

    assert state["running"] is False
    assert state["healthy"] is False
    assert state["pid"] is None
    assert state["local"] is True
    assert state["base_url"] == "http://localhost:8100/v1"


async def test_start_spawns_process_and_waits(server, monkeypatch):
    spawned = []

    class _FakeProcess:
        pid = os.getpid()  # 用当前进程冒充，避免 _wait_healthy 判定「进程已退出」

    def fake_popen(cmd, **kwargs):
        spawned.append((cmd, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(rerank_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_find_process", lambda: None)
    # 第一次（start 入口）不健康，spawn 之后变健康
    _patch_health(monkeypatch, server, [False, True])

    state = await server.start(wait=True)

    assert spawned, "应该拉起子进程"
    cmd, kwargs = spawned[0]
    assert cmd[1] == "serve"
    assert kwargs["start_new_session"] is True
    assert state["healthy"] is True
    assert state["pid"] == os.getpid()
    assert json.loads(server._pid_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


async def test_start_is_idempotent_when_healthy(server, monkeypatch):
    def fail_popen(*args, **kwargs):
        raise AssertionError("已在运行就不应再拉起进程")

    monkeypatch.setattr(rerank_server.subprocess, "Popen", fail_popen)
    _patch_health(monkeypatch, server, [True])

    state = await server.start(wait=False)

    assert state["message"] == "已经在运行"


async def test_start_rejects_remote_base_url(server, monkeypatch):
    monkeypatch.setattr(
        rerank_server.get_settings(), "rerank_base_url", "http://gpu-box:8100/v1"
    )
    with pytest.raises(RerankServerError):
        await server.start(wait=False)


async def test_start_rejects_non_http_backend(server, monkeypatch):
    monkeypatch.setattr(rerank_server.get_settings(), "rerank_backend", "flashrank")
    with pytest.raises(RerankServerError):
        await server.start(wait=False)


async def test_stop_terminates_managed_process(server, monkeypatch):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        server._write_pid(process.pid, ["fake"])
        _patch_health(monkeypatch, server, [False])
        monkeypatch.setattr(server, "_find_process", lambda: None)

        state = await server.stop()

        process.wait(timeout=10)
        assert process.poll() is not None
        assert state["running"] is False
        assert not server._pid_path.exists()
    finally:
        if process.poll() is None:
            process.kill()


async def test_stop_without_service(server, monkeypatch):
    _patch_health(monkeypatch, server, [False])
    monkeypatch.setattr(server, "_find_process", lambda: None)

    state = await server.stop()

    assert state["message"] == "没有发现运行中的 rerank 服务"


def test_find_process_matches_model_and_port(server, monkeypatch):
    argv_list = [
        (111, ["/home/will/llm/bin/vllm", "serve", "other-model", "--port", "8100"]),
        (222, ["/home/will/llm/bin/vllm", "serve", "BAAI/bge-reranker-v2-m3", "--port", "8002"]),
        (
            333,
            [
                "/home/will/llm/bin/vllm",
                "serve",
                "BAAI/bge-reranker-v2-m3",
                "--port",
                "8100",
                "--host",
                "0.0.0.0",
            ],
        ),
        (444, ["python", "something.py", "--port", "8100"]),
    ]
    monkeypatch.setattr(rerank_server, "_iter_proc_cmdlines", lambda: iter(argv_list))

    assert server._find_process() == 333


def test_tail_log_missing_file(server):
    assert server.tail_log(10) == []


def test_is_local_accepts_machine_ip(server, monkeypatch):
    """base_url 写本机网卡 IP 时仍视为本机（WSL 下 loopback 不通，配置里常写 IP）。"""
    ip = rerank_server._primary_ip()
    assert ip in rerank_server._interface_ips() | rerank_server._LOCAL_HOSTS
    monkeypatch.setattr(
        rerank_server.get_settings(), "rerank_base_url", f"http://{ip}:8100/v1"
    )
    assert server.is_local() is True


def test_is_local_rejects_remote_host(server, monkeypatch):
    monkeypatch.setattr(
        rerank_server.get_settings(), "rerank_base_url", "http://gpu-box.example:8100/v1"
    )
    assert server.is_local() is False


def test_storage_paths_are_absolute_independent_of_cwd(server, tmp_path):
    """相对 storage_root 按项目根解析，CLI 从 backend/ 启动也落到同一处。"""
    assert server._log_path.is_absolute()
    assert server._pid_path.is_absolute()
    assert str(server._log_path).startswith(str(tmp_path))
