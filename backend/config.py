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

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # 存储
    storage_root: str = "./storage"

    # Doubao / Ark
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

    # 后台任务
    # 启动时把遗留的 pending/processing 文档标记为 failed
    cleanup_stale_documents: bool = True
    # 心跳超过该秒数未更新，视为任务已死（防止进程活着但卡死）
    stale_task_grace_seconds: int = 300
    # 摄入任务心跳写盘间隔（秒）
    task_heartbeat_interval: int = 15

    # 多轮对话：检索前把依赖上文的追问改写成可独立检索的问题
    # llm = 用 LLM 改写；concat = 仅拼接上一轮问题（不额外调用模型）；off = 关闭
    query_rewrite: str = "llm"
    query_rewrite_max_tokens: int = 2048

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
