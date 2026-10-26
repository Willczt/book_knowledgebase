"""
把底层网络 / HTTP 异常翻译成可读、可操作的提示信息。

tenacity 重试失败后会抛 RetryError，openai SDK 会抛 APIConnectionError 等，
直接把这些原始异常冒到前端（例如 `RetryError[<Future ... ConnectError>]`）对用户毫无意义。
"""

from typing import Optional

import httpx
from tenacity import RetryError

from core.exceptions import EmbeddingProviderUnavailable, LLMProviderUnavailable

try:
    import openai
except ImportError:  # 理论上一定存在（依赖里已有），兜底避免导入失败
    openai = None  # type: ignore[assignment]


# 错误分类
CONNECT = "connect"       # 服务没启动 / 端口不通 / DNS 失败
TIMEOUT = "timeout"       # 超时
AUTH = "auth"             # 401/403，Key 无效
NOT_FOUND = "not_found"   # 404，模型不存在
RATE_LIMIT = "rate_limit"  # 429
BAD_REQUEST = "bad_request"  # 其它 4xx
SERVER = "server"         # 5xx
UNKNOWN = "unknown"


def unwrap(exc: Exception) -> Exception:
    """剥掉 tenacity 的 RetryError，取出真正的原因"""
    if isinstance(exc, RetryError):
        last_attempt = getattr(exc, "last_attempt", None)
        if last_attempt is not None:
            inner = last_attempt.exception()
            if inner is not None:
                return inner
    return exc


def _status_code(exc: Exception) -> Optional[int]:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if openai is not None and isinstance(exc, openai.APIStatusError):
        return exc.status_code
    return None


def _response_snippet(exc: Exception, limit: int = 200) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        return (response.text or "").strip()[:limit]
    except Exception:
        return ""


def classify(exc: Exception) -> str:
    exc = unwrap(exc)

    status = _status_code(exc)
    if status is not None:
        if status in (401, 403):
            return AUTH
        if status == 404:
            return NOT_FOUND
        if status == 429:
            return RATE_LIMIT
        if status >= 500:
            return SERVER
        if 400 <= status < 500:
            return BAD_REQUEST

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return CONNECT
    if isinstance(exc, httpx.TimeoutException):
        return TIMEOUT

    if openai is not None:
        if isinstance(exc, openai.APIConnectionError):
            return CONNECT
        if isinstance(exc, openai.APITimeoutError):
            return TIMEOUT
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return AUTH
        if isinstance(exc, openai.NotFoundError):
            return NOT_FOUND
        if isinstance(exc, openai.RateLimitError):
            return RATE_LIMIT
        if isinstance(exc, openai.InternalServerError):
            return SERVER

    return UNKNOWN


def is_retryable(exc: Exception) -> bool:
    """只对「服务端临时故障 / 限流」重试；服务没启动、Key 错误重试也没用"""
    return classify(exc) in (SERVER, RATE_LIMIT)


def _hint(provider: str, base_url: str, model_id: str, timeout: float, snippet: str) -> dict:
    """按 provider 给出对应的排查建议"""
    if provider == "ollama":
        return {
            CONNECT: (
                f"无法连接到本地 Ollama（{base_url}）。"
                "请确认 Ollama 已启动（在终端运行 `ollama serve`），"
                "并检查 OLLAMA_BASE_URL 是否正确。"
            ),
            TIMEOUT: (
                f"Ollama 响应超时（超过 {timeout:.0f}s）。"
                "可调大 EMBEDDING_TIMEOUT，或调小批次大小。"
            ),
            NOT_FOUND: (
                f"Ollama 中找不到模型 {model_id}，请先执行 `ollama pull {model_id}`。"
            ),
            AUTH: "Ollama 返回鉴权错误，请确认该实例是否需要 API Key。",
            RATE_LIMIT: "Ollama 请求过于频繁（429），请稍后重试。",
            SERVER: f"Ollama 服务内部错误（HTTP 5xx）：{snippet}",
            BAD_REQUEST: f"Ollama 拒绝了请求（参数或模型不兼容）：{snippet}",
        }
    if provider == "doubao":
        return {
            CONNECT: f"无法连接豆包 Embedding 服务（{base_url}）。请检查网络或代理设置。",
            TIMEOUT: f"豆包 Embedding 响应超时（超过 {timeout:.0f}s）。可调大 EMBEDDING_TIMEOUT。",
            AUTH: "豆包 API Key 无效或已过期，请检查 .env 中的 ARK_API_KEY。",
            NOT_FOUND: f"豆包 Embedding 模型 {model_id} 不存在或无权限，请检查模型 ID 配置。",
            RATE_LIMIT: "豆包接口触发限流（429），请稍后重试或降低并发。",
            SERVER: f"豆包服务异常（HTTP 5xx）：{snippet}",
            BAD_REQUEST: f"豆包拒绝了请求（参数或模型不兼容）：{snippet}",
        }
    if provider == "vllm":
        return {
            CONNECT: f"无法连接 vLLM 服务（{base_url}），请确认已启动 `vllm serve`。",
            TIMEOUT: f"vLLM 响应超时（超过 {timeout:.0f}s）。可调大 EMBEDDING_TIMEOUT。",
            AUTH: "vLLM 鉴权失败，请检查 VLLM_API_KEY。",
            NOT_FOUND: f"vLLM 中找不到模型 {model_id}，请检查启动参数里的 --model。",
            SERVER: f"vLLM 服务异常（HTTP 5xx）：{snippet}",
            BAD_REQUEST: f"vLLM 拒绝了请求：{snippet}",
        }
    # deepseek / 通用 LLM
    return {
        CONNECT: f"无法连接 {provider} API（{base_url}）。请检查网络/代理，或确认 base_url 配置正确。",
        TIMEOUT: f"{provider} 响应超时（超过 {timeout:.0f}s）。可调大 LLM_TIMEOUT 或稍后重试。",
        AUTH: f"{provider} API Key 无效或已过期，请检查 .env 中的配置。",
        NOT_FOUND: f"{provider} 中找不到模型 {model_id}，请检查模型 ID 配置。",
        RATE_LIMIT: f"{provider} 接口触发限流（429），请稍后重试。",
        SERVER: f"{provider} 服务异常（HTTP 5xx）：{snippet}",
        BAD_REQUEST: f"{provider} 拒绝了请求：{snippet}",
    }


def friendly_embedding_error(
    exc: Exception,
    *,
    provider: str,
    base_url: str,
    model_id: str,
    timeout: float,
) -> Exception:
    """返回可直接展示给用户的异常对象"""
    reason = classify(exc)
    if reason == UNKNOWN:
        return exc

    message = _hint(provider, base_url, model_id, timeout, _response_snippet(exc)).get(reason)
    if not message:
        return exc
    return EmbeddingProviderUnavailable(message)


def friendly_llm_error(
    exc: Exception,
    *,
    provider: str,
    base_url: str,
    model_id: str,
    timeout: float,
) -> Exception:
    reason = classify(exc)
    if reason == UNKNOWN:
        return exc

    if provider == "deepseek" and reason == BAD_REQUEST:
        snippet = _response_snippet(exc)
        if "insufficient balance" in snippet.lower() or "余额" in snippet:
            return LLMProviderUnavailable("DeepSeek 账户余额不足，请充值后再试。")

    message = _hint(provider, base_url, model_id, timeout, _response_snippet(exc)).get(reason)
    if not message:
        return exc
    return LLMProviderUnavailable(message)
