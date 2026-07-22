import asyncio
from datetime import datetime, timezone

import structlog

from config import get_settings
from core.embedding.registry import get_embedding_registry
from core.indexing.indexer import DocumentIndexer
from vector_store.qdrant_store import QdrantVectorStore

logger = structlog.get_logger()
settings = get_settings()


async def run_ingest_task(
    doc_id: str,
    file_path: str,
    source_doc: str,
    kb_id: str,
    kb_embedding_provider: str,
    kb_embedding_model_id: str,
    kb_embedding_vector_dim: int,
    kb_qdrant_collection: str,
    caption_provider: str | None = None,
    caption_model_id: str | None = None,
) -> None:
    """
    异步后台摄入任务。
    由 API 层通过 asyncio.create_task() 调用。
    """
    from db.engine import AsyncSessionLocal
    from db import crud

    async with AsyncSessionLocal() as db:
        doc = await crud.get_document(db, doc_id)
        if doc is None:
            logger.error("ingest_task_doc_not_found", doc_id=doc_id)
            return

        # 更新状态
        await crud.update_document(db, doc, {
            "status": "processing",
            "processing_started_at": datetime.now(timezone.utc),
        })
        logger.info("🚀 摄入任务开始", doc_id=doc_id, source_doc=source_doc, kb_id=kb_id,
                    embedding_model=kb_embedding_model_id)

    try:
        # 获取 embedding provider
        registry = get_embedding_registry()
        try:
            embedding_provider = registry.get(kb_embedding_provider, kb_embedding_model_id)
        except KeyError:
            # 动态注册（用于自定义 Ollama/vLLM 模型）
            embedding_provider = registry.register_dynamic(
                kb_embedding_provider,
                kb_embedding_model_id,
                kb_embedding_vector_dim,
            )

        vector_store = QdrantVectorStore()

        # 创建一个临时 KB 对象供 Indexer 使用
        class _KB:
            id = kb_id
            qdrant_collection_name = kb_qdrant_collection

        indexer = DocumentIndexer(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb=_KB(),
            storage_root=settings.storage_root,
        )

        chunk_count, image_count = await indexer.index_document(
            file_path=file_path,
            doc_id=doc_id,
            source_doc=source_doc,
        )

        # 更新状态为成功
        async with AsyncSessionLocal() as db:
            doc = await crud.get_document(db, doc_id)
            if doc:
                await crud.update_document(db, doc, {
                    "status": "indexed",
                    "chunk_count": chunk_count,
                    "image_count": image_count,
                    "processing_completed_at": datetime.now(timezone.utc),
                    "error_message": None,
                })

        await vector_store.close()

    except Exception as e:
        logger.error("ingest_task_failed", doc_id=doc_id, error=str(e), exc_info=True)
        async with AsyncSessionLocal() as db:
            doc = await crud.get_document(db, doc_id)
            if doc:
                await crud.update_document(db, doc, {
                    "status": "failed",
                    "error_message": str(e)[:2000],
                    "processing_completed_at": datetime.now(timezone.utc),
                })
