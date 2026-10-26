"""请求上下文中间件：请求 ID 透传、访问日志上下文、可选 Token 鉴权。"""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from core.middleware import RequestContextMiddleware


async def _ok(request):
    return JSONResponse({"ok": True})


def _client(api_token: str | None = None) -> TestClient:
    app = Starlette(routes=[Route("/api/v1/x", _ok), Route("/healthz", _ok)])
    app.add_middleware(RequestContextMiddleware, api_token=api_token)
    return TestClient(app)


def test_request_id_is_generated_and_returned():
    response = _client().get("/api/v1/x")
    assert response.status_code == 200
    assert len(response.headers["x-request-id"]) == 16


def test_request_id_is_passed_through():
    response = _client().get("/api/v1/x", headers={"X-Request-ID": "trace-abc"})
    assert response.headers["x-request-id"] == "trace-abc"


def test_no_token_configured_keeps_api_open():
    assert _client().get("/api/v1/x").status_code == 200


def test_token_enforced_when_configured():
    client = _client("s3cret")
    denied = client.get("/api/v1/x")
    assert denied.status_code == 401
    assert denied.json()["request_id"]

    assert client.get("/api/v1/x", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/v1/x", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_healthz_is_public_even_with_token():
    assert _client("s3cret").get("/healthz").status_code == 200
