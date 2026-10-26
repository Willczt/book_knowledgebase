"""日志配置：开发用可读控制台，生产用单行 JSON（便于 Loki/ELK 采集）。

生产环境设置 `LOG_FORMAT=json`，所有 structlog 事件都会带 request_id / path 等
上下文字段，配合中间件生成的 `X-Request-ID` 可以在网关日志里串起整条请求。
"""

import logging
import sys

import structlog

from config import get_settings

_NOISY_LOGGERS = ("httpx", "httpcore", "qdrant_client", "urllib3")


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    json_mode = settings.log_format.lower() == "json"

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_mode:
        processors = shared + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
    else:
        processors = shared + [
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
