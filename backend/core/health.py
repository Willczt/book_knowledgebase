"""
外部模型服务的连通性探测（供 /health 使用）。

探测本身必须快且不能抛异常，超时也控制在几秒内，避免健康检查被拖死。
"""

import time

import httpx

from config import get_settings

settings = get_settings()


def _fail(url: str, exc: Exception, started: float) -> dict:
    return {
        "ok": False,
        "url": url,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "error": _describe(exc),
    }


def _describe(exc: Exception) -> str:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "连接被拒绝（服务未启动或端口不通）"
    if isinstance(exc, httpx.TimeoutException):
        return "请求超时"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return f"鉴权失败（HTTP {code}）"
        return f"HTTP {code}: {(exc.response.text or '')[:120]}"
    return str(exc)[:160]


async def check_ollama(timeout: float = 3.0) -> dict:
    """Ollama 是否可达，以及本地有哪些可用的 embedding / 视觉模型"""
    url = settings.ollama_base_url.rstrip("/")
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/api/tags")
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
    except Exception as exc:
        return _fail(url, exc, started)

    vision = settings.ollama_vision_model
    return {
        "ok": True,
        "url": url,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "model_count": len(models),
        "has_embedding_model": any(
            name.split(":")[0] in (settings.ollama_default_embed_model, "bge-m3", "nomic-embed-text", "mxbai-embed-large")
            for name in models
        ),
        "has_vision_model": any(
            name == vision or name.startswith(f"{vision}:") for name in models
        ),
    }


async def check_deepseek(timeout: float = 5.0) -> dict:
    """DeepSeek 是否可达、Key 是否有效，以及默认模型的能力"""
    url = settings.deepseek_base_url.rstrip("/")
    started = time.perf_counter()

    if not settings.deepseek_api_key:
        return {
            "ok": False,
            "url": url,
            "configured": False,
            "latency_ms": 0,
            "error": "未配置 DEEPSEEK_API_KEY",
        }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{url}/models",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
    except Exception as exc:
        return {**_fail(url, exc, started), "configured": True}

    model = next((m for m in data if m.get("id") == settings.deepseek_model), None)
    return {
        "ok": True,
        "url": url,
        "configured": True,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "model": settings.deepseek_model,
        "model_exists": model is not None,
        "supports_vision": bool(model and "image" in (model.get("input_modalities") or [])),
    }

