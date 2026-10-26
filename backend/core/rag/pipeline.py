import base64
from typing import AsyncIterator, Optional

import structlog

from config import get_settings
from core.exceptions import LLMEmptyResponse
from core.llm.base import ChatMessage, LLMProvider, MessageRole
from core.rag.query_rewriter import rewrite_query
from core.retrieval.retriever import HybridRetriever
from db.models import KnowledgeBase
from db.schemas import SourceChunk
from vector_store.qdrant_store import SearchResult

logger = structlog.get_logger()
settings = get_settings()

_SYSTEM_PROMPT = """你是一个专业的知识库问答助手，根据提供的参考资料回答用户问题。

回答要求：
1. 只基于提供的参考资料回答，不要凭空捏造信息
2. 如果参考资料中没有相关信息，请明确告知用户，不要拿无关段落凑答案
3. 回答要准确、简洁、有条理
4. 如果有多个来源，请综合信息给出完整回答
5. 适当引用来源文档名称以增加可信度
6. 输出的内容带序号，markdown格式优先
7. 只回答【当前问题】。历史消息只用于理解指代关系（如"他""这件事"），
   不要重复回答历史消息里的问题，也不要把历史问答当成需要输出的内容
"""


def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "（未找到相关参考资料）"

    parts = []
    for i, r in enumerate(results, 1):
        source_info = f"来源：{r.source_doc}"
        if r.page_num:
            source_info += f"，第 {r.page_num} 页"
        if r.chunk_type == "image_caption":
            parts.append(f"[参考资料 {i} - 图片描述]\n{source_info}\n{r.text}")
        else:
            parts.append(f"[参考资料 {i}]\n{source_info}\n{r.text}")

    return "\n\n---\n\n".join(parts)


def _build_history_messages(history: list) -> list[ChatMessage]:
    """
    将数据库 Message 对象列表转换为 ChatMessage 列表。

    只保留「问 + 有内容的答」成对的完整轮次：
    - 空的 assistant 消息（模型没产出）会连同它对应的提问一起丢弃，
      否则模型会把它当成一个待补全的空回答，转而去回答上一轮的提问；
    - 落单的 user 消息（没有对应回答）同样丢弃。
    """
    messages: list[ChatMessage] = []
    pending_user: Optional[str] = None

    for msg in history:
        if msg.role == "user":
            # 上一条提问没等到回答，直接丢弃
            pending_user = msg.content
            continue

        if pending_user is None:
            # 孤立的 assistant 消息，跳过
            continue
        if not (msg.content or "").strip():
            # 空回答：整轮丢弃
            pending_user = None
            continue

        messages.append(ChatMessage(role=MessageRole.user, content=pending_user))
        messages.append(ChatMessage(role=MessageRole.assistant, content=msg.content))
        pending_user = None

    return messages


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_provider: LLMProvider,
        kb: KnowledgeBase,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_provider
        self._kb = kb
        self._rewrite_llm: Optional[LLMProvider] = None
        self._rewrite_llm_ready = False

    def _get_rewrite_llm(self) -> Optional[LLMProvider]:
        """
        改写用独立的 LLM 实例：温度 0、限制输出长度、推理强度 low，
        避免推理模型把思考算满额度（那会导致空输出），同时压低延迟。
        """
        if self._rewrite_llm_ready:
            return self._rewrite_llm

        self._rewrite_llm_ready = True
        cfg = self._llm.config
        try:
            from core.llm.registry import create_llm_provider

            self._rewrite_llm = create_llm_provider(
                cfg.provider,
                cfg.model_id,
                {
                    "max_tokens": settings.query_rewrite_max_tokens,
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                    "timeout": min(cfg.timeout, 60),
                },
            )
        except Exception as exc:
            logger.warning("query_rewrite_llm_init_failed", error=str(exc)[:200])
            self._rewrite_llm = None
        return self._rewrite_llm

    async def retrieve(
        self,
        query_text: str,
        query_image: Optional[bytes] = None,
        top_k: int = 8,
        score_threshold: float = 0.0,
        use_reranker: Optional[bool] = None,
    ) -> list[SearchResult]:
        query = await self._retriever.build_query(
            query_text=query_text,
            query_image=query_image,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return await self._retriever.retrieve(
            collection_name=self._kb.qdrant_collection_name,
            query=query,
            use_reranker=use_reranker,
        )

    async def stream_chat(
        self,
        query_text: str,
        history: list,
        query_image: Optional[bytes] = None,
        top_k: int = 8,
        score_threshold: float = 0.0,
        use_reranker: Optional[bool] = None,
    ) -> AsyncIterator[dict]:
        """
        SSE 流式问答生成器。
        yield {"token": "..."} 若干次
        yield {"done": True, "sources": [...]} 一次（结束信号）
        """
        history_messages = _build_history_messages(history)

        # 1. 检索前先补全指代："二者矛盾体现在哪里？" → "曾国藩和李鸿章的矛盾体现在哪里？"
        retrieval_query, rewrite_method = await rewrite_query(
            self._get_rewrite_llm() if settings.query_rewrite == "llm" else None,
            history_messages,
            query_text,
        )
        if rewrite_method != "none":
            logger.info(
                "query_rewritten",
                original=query_text[:60],
                rewritten=retrieval_query[:80],
                method=rewrite_method,
            )

        results = await self.retrieve(
            query_text=retrieval_query,
            query_image=query_image,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranker=use_reranker,
        )
        logger.info("rag_retrieve_done", query=retrieval_query[:60], result_count=len(results),
                    collection=self._kb.qdrant_collection_name)

        # 2. 构建 prompt
        context = _format_context(results)

        system_with_context = _SYSTEM_PROMPT + f"\n\n参考资料：\n{context}"

        messages = [
            ChatMessage(role=MessageRole.system, content=system_with_context),
            *history_messages,
        ]

        # 如果用户附带了图片，构造多模态 user 消息
        if query_image:
            b64 = base64.b64encode(query_image).decode()
            user_content = [
                {"type": "text", "text": query_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
            messages.append(ChatMessage(role=MessageRole.user, content=user_content))
        else:
            messages.append(ChatMessage(role=MessageRole.user, content=query_text))

        # 3. 流式生成
        produced = 0
        async for token in self._llm.stream_chat(messages):
            if token:
                produced += 1
                yield {"token": token}

        if produced == 0:
            raise LLMEmptyResponse("模型没有返回任何内容，请重试。")

        # 4. 返回 sources
        sources = [
            SourceChunk(
                chunk_id=r.chunk_id,
                text=r.text[:300],  # 截断避免响应过大
                # 开了重排就用 rerank 分展示，顺序与百分比才一致
                score=round(r.rerank_score if r.rerank_score is not None else r.score, 4),
                source_doc=r.source_doc,
                doc_id=r.doc_id,
                page_num=r.page_num,
                chunk_type=r.chunk_type,
                image_path=r.image_path,
            ).model_dump()
            for r in results
        ]
        yield {
            "done": True,
            "sources": sources,
            # 便于前端/调试查看实际用于检索的查询
            "rewritten_query": retrieval_query if rewrite_method != "none" else None,
        }
