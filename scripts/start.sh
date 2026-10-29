#!/usr/bin/env bash
# 一键启动开发环境：Ollama（Windows）→ rerank 模型 → 后端 → 前端
#
# 用法：
#   ./scripts/start.sh                  # 完整启动
#   ./scripts/start.sh --no-frontend    # 只起 rerank + 后端
#   ./scripts/start.sh --no-rerank      # 跳过 rerank（精排自动降级为融合分）
#   ./scripts/start.sh --strict-rerank  # rerank 起不来就退出，不启动后端
#   ./scripts/start.sh --port 8001      # 换后端端口
#
# 说明：
# - Ollama 由 Windows 提供（WSL 里没有 ollama 可执行文件），脚本会自动探测
#   localhost / 默认网关 / WSL 的 DNS 服务器，并把 OLLAMA_BASE_URL 传给后端；
# - 对话默认用本地模型（前端取 /llm-providers 的第一项 = ollama），
#   本地模型不可用时后端自动兜底到 DEEPSEEK（LLM_FALLBACK_PROVIDER）；
# - 后端从仓库根目录启动（PYTHONPATH=backend），否则相对路径的 DATABASE_URL
#   会指向 backend/storage/app.db，等于换了一个空库。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

PORT=8000
FRONTEND_PORT=5173
START_RERANK=1
START_FRONTEND=1
STRICT_RERANK=0
RELOAD=1

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-rerank) START_RERANK=0; shift ;;
    --no-frontend) START_FRONTEND=0; shift ;;
    --strict-rerank) STRICT_RERANK=1; shift ;;
    --no-reload) RELOAD=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1（--help 查看用法）"; exit 2 ;;
  esac
done

BACKEND_PID=""
FRONTEND_PID=""

info() { printf '[..] %s\n' "$*"; }
ok()   { printf '[ok] %s\n' "$*"; }
warn() { printf '[!!] %s\n' "$*"; }
fail() { printf '[xx] %s\n' "$*"; }

cleanup() {
  echo
  info "停止本次启动的进程..."
  # 进程组一起杀：uvicorn --reload 和 npm 都会再派生子进程
  [ -n "$BACKEND_PID" ] && kill -- -"$BACKEND_PID" 2>/dev/null && ok "后端已停止 (pgid $BACKEND_PID)"
  [ -n "$FRONTEND_PID" ] && kill -- -"$FRONTEND_PID" 2>/dev/null && ok "前端已停止 (pgid $FRONTEND_PID)"
  rm -f "$LOG_DIR/frontend.pid"
  echo "rerank 服务未停止；需要时执行：make rerank-down"
  exit 0
}
trap cleanup INT TERM

# .env 里取配置（不 source，避免执行任意内容）
env_get() {
  local key="$1" default="${2:-}" value
  [ -f "$ROOT/.env" ] || { printf '%s' "$default"; return; }
  value="$(grep -E "^${key}=" "$ROOT/.env" | tail -1 | cut -d= -f2-)"
  value="${value%$'\r'}"
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  [ -n "$value" ] && printf '%s' "$value" || printf '%s' "$default"
}

url_ok() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }

wait_for() {  # wait_for <url> <名称> <超时秒>
  local url="$1" name="$2" timeout="${3:-60}" i
  for ((i = 1; i <= timeout; i++)); do
    url_ok "$url" && return 0
    sleep 1
  done
  fail "$name 在 ${timeout}s 内未就绪（$url）"
  return 1
}

# ---------- 1. Ollama（Windows 提供） ----------
info "探测 Ollama ..."
OLLAMA_CANDIDATES=()
[ -n "${OLLAMA_BASE_URL:-}" ] && OLLAMA_CANDIDATES+=("${OLLAMA_BASE_URL%/}")
OLLAMA_CANDIDATES+=("http://127.0.0.1:11434")

GATEWAY="$(ip route 2>/dev/null | awk '/^default/{print $3; exit}')"
[ -n "${GATEWAY:-}" ] && OLLAMA_CANDIDATES+=("http://${GATEWAY}:11434")

WSL_DNS="$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"
[ -n "${WSL_DNS:-}" ] && [ "$WSL_DNS" != "127.0.0.53" ] && OLLAMA_CANDIDATES+=("http://${WSL_DNS}:11434")

OLLAMA_CANDIDATES+=("http://host.docker.internal:11434")

OLLAMA_URL=""
for candidate in "${OLLAMA_CANDIDATES[@]}"; do
  if url_ok "${candidate}/api/tags"; then OLLAMA_URL="$candidate"; break; fi
done

EMBED_MODEL="$(env_get OLLAMA_DEFAULT_EMBED_MODEL bge-m3)"
CHAT_MODEL="$(env_get OLLAMA_DEFAULT_CHAT_MODEL qwen3:8b)"

if [ -n "$OLLAMA_URL" ]; then
  export OLLAMA_BASE_URL="$OLLAMA_URL"
  ok "Ollama 可用：$OLLAMA_URL"
  MODELS="$("$PY" - "$OLLAMA_URL" <<'PY' 2>/dev/null
import json, sys, urllib.request
with urllib.request.urlopen(f"{sys.argv[1]}/api/tags", timeout=5) as resp:
    data = json.load(resp)
print(" ".join(m.get("name", "") for m in data.get("models", [])))
PY
)"
  echo "     模型：${MODELS:-（空）}"
  has_model() { local want="$1"; [[ " $MODELS " == *" $want "* || " $MODELS " == *" $want:"* ]]; }
  has_model "$EMBED_MODEL" || warn "缺少 embedding 模型 $EMBED_MODEL —— 知识库检索会失败，先在 Windows 上执行：ollama pull $EMBED_MODEL"
  has_model "$CHAT_MODEL" || warn "缺少对话模型 $CHAT_MODEL —— 本地对话会失败并自动兜底到 $(env_get LLM_FALLBACK_PROVIDER deepseek)"
else
  warn "没有探测到可用的 Ollama（Windows 上的 Ollama 默认只监听 127.0.0.1，WSL 访问不到）"
  cat <<'TIP'
     在 Windows 上让它监听所有网卡后重启 Ollama：
       PowerShell:  setx OLLAMA_HOST 0.0.0.0:11434
       防火墙放行 11434，然后重开 Ollama
     或手动指定：export OLLAMA_BASE_URL=http://<Windows_IP>:11434
     embedding 不可用意味着检索/问答会报错；对话侧会自动兜底到 deepseek。
TIP
fi

# ---------- 2. rerank 模型 ----------
if [ "$START_RERANK" -eq 1 ]; then
  RERANK_WAIT="$(env_get RERANK_START_TIMEOUT 600)"
  info "启动 rerank 服务（make rerank-up，最多等 ${RERANK_WAIT%.*}s），首次加载模型可能较久 ..."
  # 外层再套一层超时：vLLM 卡在模型下载/显存不足时 make 会一直等
  if timeout "${RERANK_WAIT%.*}" make rerank-up 2>&1 | tee "$LOG_DIR/rerank-up.log"; then
    RERANK_BASE="$(env_get RERANK_BASE_URL http://127.0.0.1:8100/v1)"
    RERANK_ROOT="${RERANK_BASE%/v1}"
    RERANK_ROOT="${RERANK_ROOT%/}"
    if wait_for "${RERANK_ROOT}/health" "rerank 服务" 120; then
      ok "rerank 就绪：$RERANK_BASE"
    else
      warn "rerank 健康检查未通过，精排会自动降级为融合分（不影响启动）"
      [ "$STRICT_RERANK" -eq 1 ] && { fail "--strict-rerank 已开启，退出"; exit 1; }
    fi
  else
    warn "make rerank-up 失败（远程 RERANK_BASE_URL 会拒绝本机启动；vllm 未装也会失败）"
    tail -5 "$LOG_DIR/rerank-up.log" 2>/dev/null | sed 's/^/     /'
    [ "$STRICT_RERANK" -eq 1 ] && { fail "--strict-rerank 已开启，退出"; exit 1; }
  fi
fi

# ---------- 3. 后端 ----------
info "启动后端 ..."
if url_ok "http://127.0.0.1:${PORT}/api/v1/health"; then
  ok "后端已在 ${PORT} 端口运行，跳过启动"
  warn "已运行的后端不会重新加载本次探测到的 OLLAMA_BASE_URL，需要的话先停掉它再跑本脚本"
else
  BACKEND_ARGS=(--host 0.0.0.0 --port "$PORT")
  [ "$RELOAD" -eq 1 ] && BACKEND_ARGS+=(--reload)
  PYTHONPATH="$ROOT/backend" setsid nohup "$PY" -m uvicorn main:app "${BACKEND_ARGS[@]}" \
    >> "$LOG_DIR/backend.log" 2>&1 &
  BACKEND_PID=$!
  if wait_for "http://127.0.0.1:${PORT}/api/v1/health" "后端" 90; then
    ok "后端就绪：http://localhost:${PORT}（pid $BACKEND_PID，日志 logs/backend.log）"
  else
    warn "后端启动失败，最后 20 行日志："
    tail -20 "$LOG_DIR/backend.log" 2>/dev/null | sed 's/^/     /'
    exit 1
  fi
fi

# ---------- 4. 前端 ----------
if [ "$START_FRONTEND" -eq 1 ]; then
  info "启动前端 ..."
  if url_ok "http://127.0.0.1:${FRONTEND_PORT}/"; then
    ok "前端已在 ${FRONTEND_PORT} 端口运行，跳过启动"
  else
    ( cd "$ROOT/frontend" && setsid nohup npm run dev >> "$LOG_DIR/frontend.log" 2>&1 & echo $! > "$LOG_DIR/frontend.pid" )
    sleep 1
    FRONTEND_PID="$(cat "$LOG_DIR/frontend.pid" 2>/dev/null || true)"
    if wait_for "http://127.0.0.1:${FRONTEND_PORT}/" "前端" 90; then
      ok "前端就绪：http://localhost:${FRONTEND_PORT}（日志 logs/frontend.log）"
    else
      warn "前端启动失败，最后 20 行日志："
      tail -20 "$LOG_DIR/frontend.log" 2>/dev/null | sed 's/^/     /'
    fi
  fi
fi

# ---------- 5. 汇总 ----------
echo
echo "==================== 启动完成 ===================="
echo "  前端     http://localhost:${FRONTEND_PORT}"
echo "  后端     http://localhost:${PORT}    API 文档 /docs"
echo "  Ollama   ${OLLAMA_URL:-未连接}"
echo "  对话模型 本地 ${CHAT_MODEL} → 失败兜底 $(env_get LLM_FALLBACK_PROVIDER deepseek)/$(env_get DEEPSEEK_MODEL deepseek-chat)"
echo "  rerank   $(env_get RERANK_BASE_URL 未配置)"
echo "  日志     logs/backend.log · logs/frontend.log · logs/rerank-up.log"
echo "  停止     Ctrl+C（会停掉本次启动的前后端；rerank 用 make rerank-down）"
echo "================================================="

if [ -n "$BACKEND_PID" ] || [ -n "$FRONTEND_PID" ]; then
  wait
fi
