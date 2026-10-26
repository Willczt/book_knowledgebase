"""请求上下文中间件（纯 ASGI，避免 BaseHTTPMiddleware 影响 SSE 流式输出）。

职责：
- 生成/透传 X-Request-ID，并绑定到 structlog 上下文
- 记录访问日志（方法、路径、状态码、耗时），SSE 请求耗时即整个流的时间
- 可选 API Token 校验（配置 `API_TOKEN` 后生效，OPTIONS 放行给 CORS）
"""

from __future__ import annotations

import secrets
import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = structlog.get_logger()

# 不需要鉴权的路径：健康探针、文档、OpenAPI
_PUBLIC_PREFIXES = ("/healthz", "/readyz", "/docs", "/redoc", "/openapi.json")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, api_token: str | None = None) -> None:
        self.app = app
        self.api_token = api_token.strip() if api_token else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:16]

        if self._unauthorized(method, path, headers):
            response = JSONResponse(
                {"detail": "未授权：请求缺少有效的 API Token", "request_id": request_id},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "http_request_failed",
                method=method,
                path=path,
                request_id=request_id,
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            log = logger.warning if status_code >= 500 else logger.info
            log(
                "http_request",
                method=method,
                path=path,
                status=status_code,
                duration_ms=duration_ms,
                client=scope.get("client", ("", 0))[0] if scope.get("client") else None,
            )
            structlog.contextvars.clear_contextvars()

    def _unauthorized(self, method: str, path: str, headers: Headers) -> bool:
        if not self.api_token:
            return False
        if method == "OPTIONS":
            return False
        if not path.startswith("/api/") and not path.startswith("/images/"):
            return False
        if any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
            return False
        provided = headers.get("authorization", "")
        return not secrets.compare_digest(provided, f"Bearer {self.api_token}")
