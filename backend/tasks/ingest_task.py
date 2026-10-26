"""文档摄入：解析 → 分块 → 向量化 → 写入 Qdrant。

任务只接受 doc_id，其余信息在任务内部从数据库读取。这样 API 进程和独立 worker
走的是同一条路径，重试时也能拿到最新的文档/知识库配置。

失败一律抛异常交给 worker 决定重试或落终态；本模块只负责清理本次写入的半截向量。
"""

import structlog

from config import get_settings
from core.embedding.registry import get_embedding_registry
from core.indexing.indexer import DocumentIndexer
from db.models import utcnow
from vector_store.qdrant_store import QdrantVectorStore

logger = structlog.get_logger()
settings = get_settings()


async def run_ingest_task(doc_id: str) -> None:
    from db import crud
    from db.engine import AsyncSessionLocal
    from tasks import progress as ingest_progress

    async with AsyncSessionLocal() as db:
        doc = await crud.get_document(db, doc_id)
        if doc is None:
            logger.info("ingest_skipped_doc_missing", doc_id=doc_id)
            return
        kb = await crud.get_kb(db, doc.kb_id)
        if kb is None:
            raise RuntimeError("知识库已被删除，无法继续摄入")

        file_path = doc.file_path
        source_doc = doc.filename
        kb_id = kb.id
        collection = kb.qdrant_collection_name
        embedding_provider_name = kb.embedding_provider
        embedding_model_id = kb.embedding_model_id
        embedding_dim = kb.embedding_vector_dim
        caption_provider = kb.caption_provider
        caption_model_id = kb.caption_model_id

    logger.info(
        "ingest_started",
        doc_id=doc_id,
        source_doc=source_doc,
        kb_id=kb_id,
        embedding_model=embedding_model_id,
    )

    vector_store = QdrantVectorStore()
    try:
        embedding_provider = await _resolve_embedding(
            embedding_provider_name, embedding_model_id, embedding_dim
        )
        caption_llm = _resolve_caption(caption_provider, caption_model_id)

        class _KB:
            id = kb_id
            qdrant_collection_name = collection

        indexer = DocumentIndexer(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb=_KB(),
            storage_root=settings.storage_root,
            caption_provider=caption_llm,
        )

        chunk_count, image_count = await indexer.index_document(
            file_path=file_path,
            doc_id=doc_id,
            source_doc=source_doc,
            progress=lambda payload: ingest_progress.update(doc_id, payload),
        )

        async with AsyncSessionLocal() as db:
            doc = await crud.get_document(db, doc_id)
            if doc is not None:
                await crud.update_document(db, doc, {
                    "status": "indexed",
                    "chunk_count": chunk_count,
                    "image_count": image_count,
                    "processing_completed_at": utcnow(),
                    "error_message": None,
                })
        logger.info(
            "ingest_succeeded",
            doc_id=doc_id,
            chunks=chunk_count,
            images=image_count,
        )
    except Exception as exc:
        logger.error(
            "ingest_failed",
            doc_id=doc_id,
            error=f"{type(exc).__name__}: {exc}"[:500],
            exc_info=True,
        )
        # 清掉本次已写入的半截向量，否则重试会产生重复 chunk
        try:
            await vector_store.delete_doc_points(collection, doc_id)
        except Exception as cleanup_error:
            logger.warning(
                "ingest_rollback_failed",
                doc_id=doc_id,
                error=str(cleanup_error)[:300],
            )
        raise
    finally:
        await vector_store.close()
        ingest_progress.finish(doc_id)


async def _resolve_embedding(provider: str, model_id: str, vector_dim: int):
    registry = get_embedding_registry()
    try:
        return registry.get(provider, model_id)
    except KeyError:
        # 动态注册（用于自定义 Ollama/vLLM 模型）
        return registry.register_dynamic(provider, model_id, vector_dim)


def _resolve_caption(provider: str | None, model_id: str | None):
    """caption 模型：知识库指定的优先，失败则回退到全局优先级。"""
    if not provider or not model_id:
        return None
    from core.llm.registry import create_llm_provider

    try:
        return create_llm_provider(provider, model_id)
    except Exception as exc:
        logger.warning(
            "kb_caption_provider_invalid",
            provider=provider,
            model_id=model_id,
            error=str(exc)[:200],
        )
        return None
