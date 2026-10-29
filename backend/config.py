from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 仓库根目录（backend/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 先读仓库根的 .env，再读当前工作目录的 .env（后者优先，便于局部覆盖）
_ENV_FILES = (str(_PROJECT_ROOT / ".env"), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 数据库
    database_url: str = "sqlite+aiosqlite:///./storage/app.db"
    # 连接池（仅 PostgreSQL 生效）
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle: int = 1800

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # 存储
    storage_root: str = "./storage"

    # 可选 provider：豆包（火山 Ark）
    # 未配置 ARK_API_KEY 时不会注册 doubao 的 LLM/Embedding，下面几个模型名保持默认即可
    ark_api_key: Optional[str] = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    doubao_chat_model: str = "doubao-seed-2-0-lite-260428"
    doubao_text_embedding_model: str = "doubao-embedding-vision-251215"
    doubao_vision_embedding_model: str = "doubao-embedding-vision-251215"
    doubao_vision_chat_model: str = "doubao-seed-2-0-lite-260428"

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_chat_model: str = "llama3"
    ollama_default_embed_model: str = "nomic-embed-text"

    # vLLM
    vllm_base_url: Optional[str] = None
    vllm_api_key: str = "EMPTY"

    # LLM 生成
    # 推理模型（deepseek-flash 等）的思考过程也计入 max_tokens，太小会导致只思考没正文
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.7
    # 推理强度（DeepSeek 等推理模型）：low 更快更省 token，留空用服务端默认
    llm_reasoning_effort: Optional[str] = None

    # LLM 兜底：本地模型（ollama / vllm）在吐出第一个 token 前失败时自动切到在线模型。
    # 已经产出内容的不切（半截回答 + 另一个模型续写会前后矛盾）。
    llm_fallback_enabled: bool = True
    llm_fallback_provider: str = "deepseek"
    # 留空则用该 provider 的默认对话模型（deepseek -> DEEPSEEK_MODEL）
    llm_fallback_model: Optional[str] = None

    # 网络超时（秒）：embedding 批量请求 / 图片 caption / 对话
    llm_timeout: int = 120
    embedding_timeout: float = 120.0
    caption_timeout: float = 60.0

    # 图片 caption（多模态：豆包 / deepseek-flash / ollama 视觉模型）
    # 显式指定时优先使用；留空则按 豆包 → DeepSeek → Ollama 自动选择
    caption_provider: Optional[str] = None
    caption_model_id: Optional[str] = None
    # caption 生成上限：推理模型（deepseek-flash）的思考也计入，太小会只有思考没有描述
    caption_max_tokens: int = 2048
    # 同时进行的 caption 请求数
    caption_concurrency: int = 5
    # Ollama 本地视觉模型名（可换成 qwen2.5vl / gemma3 等多模态模型）
    ollama_vision_model: str = "llava"

    # 后台任务（DB 队列 + 进程内 worker）
    # 同时进行的摄入任务数。Ollama 单卡建议 1~2；CPU/远程 API 可调大
    ingest_workers: int = 2
    # 失败重试次数（含首次），超过后落 failed
    ingest_max_attempts: int = 3
    # 任务租约：超时未续租视为 worker 已死，文档会被重新放回队列
    ingest_lease_seconds: int = 1800
    # 续租间隔
    ingest_lease_renew_seconds: int = 60
    # 关闭时等待在跑任务结束的最长时间
    shutdown_grace_seconds: int = 30
    # 队列空闲时的轮询间隔（用于处理 next_retry_at 到期的任务）
    ingest_poll_seconds: float = 2.0

    # 日志：console（开发）| json（生产，便于采集）
    log_level: str = "INFO"
    log_format: str = "console"

    # 可选 API 访问令牌：留空表示不校验。设置后所有 /api/* 请求需带
    # Authorization: Bearer <token>（前端用 VITE_API_TOKEN 注入）
    api_token: Optional[str] = None

    # 导入包上限：压缩包 2GB 由 MAX_IMPORT_BYTES 控制，这里防 zip bomb
    import_max_entries: int = 20000
    import_max_uncompressed_bytes: int = 8 * 1024**3

    # 多轮对话：检索前把依赖上文的追问改写成可独立检索的问题
    # llm = 用 LLM 改写；concat = 仅拼接上一轮问题（不额外调用模型）；off = 关闭
    query_rewrite: str = "llm"
    query_rewrite_max_tokens: int = 2048

    # 多跳问题拆解：把「比较 A 和 B」拆成多个子查询分别检索再融合
    # off = 关闭；auto = 只在检测到比较/并列/多主体时拆；always = 每条都让 LLM 判断
    query_decompose: str = "auto"
    query_decompose_max_subqueries: int = 3
    query_decompose_max_tokens: int = 1024

    # Rerank（检索结果二次排序）
    # 默认开启：召回 rerank_candidates 条候选，用 rerank 模型精排后截断到 top_k
    rerank_enabled: bool = True
    # http = vLLM cross-encoder 的 /v1/rerank 接口；flashrank = 本地 ONNX 模型
    rerank_backend: str = "http"
    rerank_base_url: str = "http://localhost:8100/v1"
    rerank_api_key: Optional[str] = None
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 参与精排的候选数（需 >= 最终 top_k，否则没有精排空间）
    rerank_candidates: int = 30
    rerank_timeout: float = 30.0
    # flashrank 后端使用的本地模型
    rerank_local_model: str = "ms-marco-MiniLM-L-6-v2"

    # rerank 服务（vLLM）的生命周期：由 backend/core/rerank_server.py 统一管理
    # 后端启动时是否顺带拉起 vLLM（--reload 开发模式不建议开，否则每次热重载都会重启它）
    rerank_autostart: bool = False
    # vllm 可执行文件：PATH 里的 vllm，或绝对路径（如 /home/will/llm/bin/vllm）
    rerank_server_bin: str = "vllm"
    # 监听地址：本机若存在 TUN/透明代理劫持 loopback，绑 127.0.0.1 会连不上
    rerank_server_bind: str = "0.0.0.0"
    # 追加给 `vllm serve` 的参数（按 shell 规则分词）
    rerank_server_extra_args: str = "--gpu-memory-utilization 0.3 --max-model-len 2048"
    # 启动后等待健康检查的最长秒数：WSL/冷启动下实测要 4~5 分钟才真正可用
    rerank_start_timeout: float = 600.0

    # RAG 默认参数
    default_top_k: int = 8
    context_history_turns: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_image_dimension: int = 1024

    # RAG 调试模式（全链路追踪，默认关闭，关闭时几乎零开销）
    debug_enabled: bool = False
    # 分层开关：query,retrieval,rerank,generation,performance（逗号分隔，可只开其中几层）
    debug_layers: str = "query,retrieval,rerank,generation,performance"
    # 内存里保留最近多少条 trace
    debug_max_traces: int = 100
    # 是否把 trace 落盘到 storage/traces/*.json（用于离线回放/导出）
    debug_persist: bool = True

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
