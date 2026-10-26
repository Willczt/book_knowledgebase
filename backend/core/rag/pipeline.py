import base64
import re
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import structlog

from config import get_settings
from core.exceptions import LLMEmptyResponse
from core.llm.base import ChatMessage, LLMProvider, MessageRole
from core.observability.trace import estimate_tokens, span
from core.rag.query_rewriter import needs_context, rewrite_query
from core.retrieval.retriever import HybridRetriever
from db.models import KnowledgeBase
from db.schemas import SourceChunk
from vector_store.qdrant_store import SearchResult

if TYPE_CHECKING:  # 仅类型检查，避免运行时循环导入
    from core.observability.trace import QueryTrace

logger = structlog.get_logger()
settings = get_settings()

_SENTENCE_SPLIT = re.compile(r"[。！？；!?;\n]+")


def _match_citations(answer: str, results: list[SearchResult]) -> list[dict[str, Any]]:
    """粗粒度引用判断：片段中的整句是否原样出现在回答里（不额外调用模型）。"""
    normalized_answer = re.sub(r"\s+", "", answer or "")
    citations: list[dict[str, Any]] = []
    for index, result in enumerate(results, 1):
        cited = False
        evidence: Optional[str] = None
        for sentence in _SENTENCE_SPLIT.split(result.text or ""):
            sentence = re.sub(r"\s+", "", sentence)
            if len(sentence) < 12:
                continue
            if sentence[:20] in normalized_answer:
                cited = True
                evidence = sentence[:60]
                break
        citations.append(
            {
                "ref": index,
                "chunk_id": result.chunk_id,
                "cited": cited,
                "evidence": evidence,
            }
        )
    return citations


def _rewrite_reason(query_text: str, method: str, history_turns: int) -> str:
    """解释这次有没有改写、为什么（便于判断问题是否在进入检索前就被扭曲）。"""
    if method == "llm":
        return "问题含指代或省略，LLM 改写成功"
    if method == "concat":
        return "问题含指代，LLM 不可用或改写失败，退化为拼接上一轮提问"
    if (settings.query_rewrite or "").lower() == "off":
        return "改写开关已关闭"
    if history_turns == 0:
        return "首轮对话，没有可用的历史上下文"
    if not needs_context(query_text):
        return "问题不含指代且信息完整，无需改写"
    return "无需改写"


def _dump_message(message: ChatMessage) -> dict[str, Any]:
    """把消息转成可序列化结构（多模态内容只保留文字与 [图片] 占位）。"""
    content = message.content
    if isinstance(content, str):
        text = content
    else:
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                parts.append("[图片]")
        text = "\n".join(parts)
    return {"role": message.role.value, "content": text}


def _record_generation_input(
    trace: "QueryTrace",
    messages: list[ChatMessage],
    context: str,
    results: list[SearchResult],
    llm: LLMProvider,
) -> Optional[dict[str, Any]]:
    """记录送进 LLM 的完整 Prompt、上下文片段的 Token 占用与模型参数。"""
    if not trace.on("generation"):
        return None

    chunk_rows = []
    for index, result in enumerate(results, 1):
        chunk_rows.append(
            {
                "ref": index,
                "chunk_id": result.chunk_id,
                "source_doc": result.source_doc,
                "page_num": result.page_num,
                "chunk_type": result.chunk_type,
                "tokens": estimate_tokens(result.text or ""),
                "order": index,
                "included": True,
                "truncated": False,
                "text": result.text,
            }
        )

    prompt_dump = [_dump_message(message) for message in messages]
    prompt_tokens = sum(estimate_tokens(item["content"]) for item in prompt_dump)
    context_tokens = sum(row["tokens"] for row in chunk_rows)
    config = getattr(llm, "config", None)

    trace.update(
        "generation",
        {
            "prompt": {
                "system": _SYSTEM_PROMPT,
                "messages": prompt_dump,
                "context_text": context,
                "context_ref_count": len(results),
            },
            "context_chunks": chunk_rows,
            "truncation": {
                "applied": False,
                "strategy": "none",
                "dropped": [],
                "note": "检索结果原样拼接进 Prompt，未做截断/压缩",
            },
            "params": {
                "provider": getattr(config, "provider", None),
                "model_id": getattr(config, "model_id", None),
                "temperature": getattr(config, "temperature", None),
                "top_p": getattr(config, "top_p", None),
                "max_tokens": getattr(config, "max_tokens", None),
                "reasoning_effort": getattr(config, "reasoning_effort", None),
            },
        },
    )

    usage = {
        "prompt_estimated": prompt_tokens,
        "context_estimated": context_tokens,
        "completion_estimated": 0,
        "estimated": True,
        "note": "LLM 接口未返回 usage，按字符数估算",
    }
    trace.set("generation", "tokens", usage)
    return usage

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
        rerank_candidates: Optional[int] = None,
        trace: Optional["QueryTrace"] = None,
    ) -> list[SearchResult]:
        query = await self._retriever.build_query(
            query_text=query_text,
            query_image=query_image,
            top_k=top_k,
            score_threshold=score_threshold,
            rerank_candidates=rerank_candidates,
            trace=trace,
        )
        return await self._retriever.retrieve(
            collection_name=self._kb.qdrant_collection_name,
            query=query,
            use_reranker=use_reranker,
            trace=trace,
        )

    async def stream_chat(
        self,
        query_text: str,
        history: list,
        query_image: Optional[bytes] = None,
        top_k: int = 8,
        score_threshold: float = 0.0,
        use_reranker: Optional[bool] = None,
        rerank_candidates: Optional[int] = None,
        trace: Optional["QueryTrace"] = None,
    ) -> AsyncIterator[dict]:
        """
        SSE 流式问答生成器。
        yield {"token": "..."} 若干次
        yield {"done": True, "sources": [...]} 一次（结束信号）
        """
        history_messages = _build_history_messages(history)
        history_turns = len(history_messages) // 2

        if trace is not None and trace.on("query"):
            trace.update(
                "query",
                {
                    "original": query_text,
                    "history_turns": history_turns,
                    "has_image": query_image is not None,
                },
            )

        # 1. 检索前先补全指代："二者矛盾体现在哪里？" → "曾国藩和李鸿章的矛盾体现在哪里？"
        with span(trace, "query_rewrite"):
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
        if trace is not None:
            trace.update(
                "query",
                {
                    "rewritten": retrieval_query,
                    "rewrite": {
                        "method": rewrite_method,
                        "changed": retrieval_query != query_text,
                        "mode": settings.query_rewrite,
                        "reason": _rewrite_reason(
                            query_text, rewrite_method, history_turns
                        ),
                    },
                },
            )

        results = await self.retrieve(
            query_text=retrieval_query,
            query_image=query_image,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranker=use_reranker,
            rerank_candidates=rerank_candidates,
            trace=trace,
        )
        logger.info("rag_retrieve_done", query=retrieval_query[:60], result_count=len(results),
                    collection=self._kb.qdrant_collection_name)

        # 2. 构建 prompt
        with span(trace, "prompt_build"):
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

        token_usage: Optional[dict[str, Any]] = None
        if trace is not None:
            token_usage = _record_generation_input(
                trace, messages, context, results, self._llm
            )

        # 3. 流式生成
        produced = 0
        collected: list[str] = []
        with span(trace, "llm_generate"):
            async for token in self._llm.stream_chat(messages):
                if token:
                    produced += 1
                    if trace is not None and trace.on("generation"):
                        collected.append(token)
                    yield {"token": token}

        if produced == 0:
            raise LLMEmptyResponse("模型没有返回任何内容，请重试。")

        if trace is not None and trace.on("generation"):
            answer = "".join(collected)
            trace.update(
                "generation",
                {
                    "output": answer,
                    "citations": _match_citations(answer, results),
                    "tokens": {
                        **(token_usage or {}),
                        "completion_estimated": estimate_tokens(answer),
                        "estimated": True,
                        "note": "LLM 接口未返回 usage，按字符数估算",
                    },
                },
            )

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
