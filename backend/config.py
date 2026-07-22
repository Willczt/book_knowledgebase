from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    # RAG 默认参数
    default_top_k: int = 8
    context_history_turns: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_image_dimension: int = 1024

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
