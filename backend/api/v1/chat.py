import json
import uuid
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import decode_base64_image, get_db, get_qdrant, get_session_or_404
from config import get_settings
from core.llm.registry import create_llm_provider
from core.rag.pipeline import RAGPipeline
from core.retrieval.reranker import get_reranker
from core.retrieval.retriever import HybridRetriever
from db import crud
from db.models import Session, utcnow
from db.schemas import ChatRequest
from tasks.background import spawn
from vector_store.qdrant_store import QdrantVectorStore

router = APIRouter(prefix="/sessions", tags=["Chat"])
settings = get_settings()


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

    reranker = get_reranker() if use_reranker else None
    retriever = HybridRetriever(
        vector_store=qdrant,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    pipeline = RAGPipeline(retriever=retriever, llm_provider=llm, kb=kb)

    message_id = str(uuid.uuid4())
    full_response = []
    sources_data = []
    rewritten_query: str | None = None

    async def event_generator() -> AsyncIterator[str]:
        nonlocal full_response, sources_data, rewritten_query
        error_message: str | None = None
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
                    rewritten_query = event.get("rewritten_query")
        except Exception as e:
            error_message = str(e) or type(e).__name__
            yield f"data: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"
        finally:
            # 持久化消息
            full_text = "".join(full_response)
            if not full_text.strip():
                # 没有正文：告知前端明确原因，避免出现"空气泡"
                if error_message is None:
                    yield f"data: {json.dumps({'error': '模型没有返回任何内容，请重试'}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'done': True, 'message_id': message_id, 'sources': sources_data, 'rewritten_query': rewritten_query}, ensure_ascii=False)}\n\n"
            # 将查询图片落盘，供后续消息历史展示
            saved_image_paths: list[str] = []
            if query_image:
                img_dir = Path(settings.storage_root) / "images" / "query_images"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_filename = f"{message_id}.png"
                (img_dir / img_filename).write_bytes(query_image)
                saved_image_paths = [f"query_images/{img_filename}"]
            spawn(_save_messages(
                session_id=session.id,
                message_id=message_id,
                user_content=body.content,
                assistant_content=full_text,
                sources=sources_data,
                image_paths=saved_image_paths,
            ))
            # 自动生成会话标题（仅首条消息）
            if not history:
                spawn(_auto_title_session(session.id, body.content))

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
        # 空回答不入库：否则下一轮历史里会多出一个空 assistant 消息，
        # 模型会把它当成待补全的回答，转而去回答上一轮的提问
        if assistant_content.strip():
            await crud.create_message(db, {
                "id": message_id,
                "session_id": session_id,
                "role": "assistant",
                "content": assistant_content,
                "sources": sources or None,
            })
        # 更新 session updated_at（传空字典不会触发 UPDATE，onupdate 也就不会生效）
        session = await crud.get_session_by_id(db, session_id)
        if session:
            await crud.update_session(
                db, session, {"updated_at": utcnow()}
            )


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
