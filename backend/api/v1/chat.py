import asyncio
import json
import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import (
    decode_base64_image,
    get_db,
    get_session_or_404,
    validate_kb_embedding,
)
from config import get_settings
from core.embedding.base import EmbeddingProvider
from core.llm.registry import create_llm_provider
from core.rag.pipeline import RAGPipeline
from core.retrieval.reranker import CrossEncoderReranker
from core.retrieval.retriever import HybridRetriever
from db import crud
from db.models import Session
from db.schemas import ChatRequest, ChatResponse, SourceChunk
from vector_store.qdrant_store import QdrantVectorStore
from api.deps import get_qdrant

router = APIRouter(prefix="/sessions", tags=["Chat"])
settings = get_settings()


async def _build_rag_pipeline(
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    use_reranker: bool = False,
) -> RAGPipeline:
    kb_query = await crud.get_kb
    llm = create_llm_provider(
        provider=session.llm_provider,
        model_id=session.llm_model_id,
        llm_config=session.llm_config,
    )
    reranker = CrossEncoderReranker() if use_reranker else None
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    # 需要 KB 对象
    from db.engine import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        kb = await crud.get_kb(db, session.kb_id)

    return RAGPipeline(retriever=retriever, llm_provider=llm, kb=kb)


@router.post("/{session_id}/chat/stream")
async def stream_chat(
    session: Annotated[Session, Depends(get_session_or_404)],
    body: ChatRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    """SSE 流式问答：逐 token 推送，最后推送 sources"""
    # 获取当前 KB
    kb = await crud.get_kb(db, session.kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 校验 embedding 模型
    from core.embedding.registry import get_embedding_registry
    registry = get_embedding_registry()
    try:
        try:
            embedding_provider = registry.get(kb.embedding_provider, kb.embedding_model_id)
        except KeyError:
            embedding_provider = registry.register_dynamic(
                kb.embedding_provider,
                kb.embedding_model_id,
                kb.embedding_vector_dim,
            )
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

    # 解码图片
    query_image = decode_base64_image(body.image)

    # 加载历史对话
    history = await crud.get_recent_messages(db, session.id, turns=settings.context_history_turns)

    # 创建 RAG pipeline
    llm = create_llm_provider(
        provider=session.llm_provider,
        model_id=session.llm_model_id,
        llm_config=session.llm_config,
    )
    retrieval_cfg = body.retrieval_config or {}
    top_k = getattr(retrieval_cfg, "top_k", settings.default_top_k)
    score_threshold = getattr(retrieval_cfg, "score_threshold", 0.0)
    use_reranker = getattr(retrieval_cfg, "use_reranker", False)

    reranker = CrossEncoderReranker() if use_reranker else None
    retriever = HybridRetriever(
        vector_store=qdrant,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    pipeline = RAGPipeline(retriever=retriever, llm_provider=llm, kb=kb)

    message_id = str(uuid.uuid4())
    full_response = []
    sources_data = []

    async def event_generator() -> AsyncIterator[str]:
        nonlocal full_response, sources_data
        try:
            async for event in pipeline.stream_chat(
                query_text=body.content,
                history=history,
                query_image=query_image,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranker=use_reranker,
            ):
                if "token" in event:
                    full_response.append(event["token"])
                    yield f"data: {json.dumps({'token': event['token']}, ensure_ascii=False)}\n\n"
                elif event.get("done"):
                    sources_data = event.get("sources", [])
                    yield f"data: {json.dumps({'done': True, 'message_id': message_id, 'sources': sources_data}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # 持久化消息
            full_text = "".join(full_response)
            asyncio.create_task(_save_messages(
                session_id=session.id,
                message_id=message_id,
                user_content=body.content,
                assistant_content=full_text,
                sources=sources_data,
                image_paths=[f"query_image_{message_id}.png"] if query_image else [],
            ))
            # 自动生成会话标题（仅首条消息）
            if not history:
                asyncio.create_task(_auto_title_session(session.id, body.content))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _save_messages(
    session_id: str,
    message_id: str,
    user_content: str,
    assistant_content: str,
    sources: list,
    image_paths: list[str],
) -> None:
    from db.engine import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        await crud.create_message(db, {
            "session_id": session_id,
            "role": "user",
            "content": user_content,
            "image_paths": image_paths or None,
        })
        await crud.create_message(db, {
            "id": message_id,
            "session_id": session_id,
            "role": "assistant",
            "content": assistant_content,
            "sources": sources or None,
        })
        # 更新 session updated_at
        session = await crud.get_session_by_id(db, session_id)
        if session:
            await crud.update_session(db, session, {})


async def _auto_title_session(session_id: str, first_message: str) -> None:
    """使用第一条用户消息自动生成会话标题"""
    title = first_message[:80].strip()
    if len(first_message) > 80:
        title += "..."
    from db.engine import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        session = await crud.get_session_by_id(db, session_id)
        if session and not session.title:
            await crud.update_session(db, session, {"title": title})
