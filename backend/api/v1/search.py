from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import decode_base64_image, get_kb_or_404, get_qdrant
from config import get_settings
from core.embedding.registry import get_embedding_registry
from core.observability.trace import new_trace, record_trace
from core.retrieval.reranker import get_reranker
from core.retrieval.retriever import HybridRetriever
from db.models import KnowledgeBase
from db.schemas import SearchRequest, SearchResponse, SourceChunk
from vector_store.qdrant_store import QdrantVectorStore

router = APIRouter(prefix="/knowledge-bases", tags=["Search"])
settings = get_settings()


@router.post("/{kb_id}/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    kb: Annotated[KnowledgeBase, Depends(get_kb_or_404)],
    qdrant: Annotated[QdrantVectorStore, Depends(get_qdrant)],
):
    """裸检索接口：不经过 LLM，直接返回 top-k 相关 chunk（用于调试检索质量）"""
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

    query_image = decode_base64_image(body.image)
    # 未显式指定时跟随全局开关（RERANK_ENABLED，默认开启）
    use_reranker = (
        body.use_reranker if body.use_reranker is not None else settings.rerank_enabled
    )
    reranker = get_reranker() if use_reranker else None
    retriever = HybridRetriever(
        vector_store=qdrant,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )

    # 调试模式开启时记录整条检索链路（关闭时 trace 为 None）
    trace = new_trace(body.query, kind="search")
    if trace is not None:
        trace.update(
            "query",
            {
                "original": body.query,
                "kb_id": kb.id,
                "kb_name": kb.name,
                "mode": "search",
                "rerank_backend": settings.rerank_backend if use_reranker else None,
                "rerank_model": reranker.model_name if reranker is not None else None,
            },
        )

    query = await retriever.build_query(
        query_text=body.query,
        query_image=query_image,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        rerank_candidates=body.rerank_candidates,
        trace=trace,
    )
    results = await retriever.retrieve(kb.qdrant_collection_name, query, trace=trace)
    trace_id = await record_trace(trace)

    return SearchResponse(
        query=body.query,
        trace_id=trace_id,
        results=[
            SourceChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                # 开了重排就用 rerank 分展示，顺序与百分比才一致
                score=round(r.rerank_score if r.rerank_score is not None else r.score, 4),
                source_doc=r.source_doc,
                doc_id=r.doc_id,
                page_num=r.page_num,
                chunk_type=r.chunk_type,
                image_path=r.image_path,
            )
            for r in results
        ],
    )
