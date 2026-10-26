FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl 用于健康检查；PyMuPDF 自带 mupdf，无需系统库
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独一层：只有 pyproject 变化时才重新安装
COPY pyproject.toml ./
RUN python - <<'PY'
import subprocess
import sys
import tomllib

with open("pyproject.toml", "rb") as fh:
    deps = tomllib.load(fh)["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY

COPY . .

# 非 root 运行：容器内被攻破时不能改代码/写系统目录
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/storage \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 存活探针只看进程；就绪探针（/readyz）给编排器/负载均衡用
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# 多 worker 时注意：trace/进度是进程内状态，默认单 worker；
# 需要横向扩容时把 WEB_CONCURRENCY 调大，队列本身是 DB 级的，可安全并发消费。
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1} --proxy-headers --forwarded-allow-ips='*' --timeout-graceful-shutdown 45"]
