"""
多轮对话的检索查询改写。

问题背景：检索只用当前这一句话去匹配向量库，
"二者矛盾体现在哪里？" 这种依赖上文的追问几乎没有可检索的实体词，
召回的全是无关片段，模型只能说"资料里没有"。
所以检索前需要把追问补全成可独立检索的问题，例如：
    "二者矛盾体现在哪里？"  →  "曾国藩和李鸿章的矛盾体现在哪里？"
"""

import re
from typing import Optional

import structlog

from config import get_settings
from core.llm.base import ChatMessage, LLMProvider, MessageRole

logger = structlog.get_logger()
settings = get_settings()

_REWRITE_SYSTEM = """你是一个检索查询改写器。
用户会给你一段对话历史和最新的一句话，请把最新这句话改写成**不依赖上下文、可独立检索**的完整问题。

要求：
1. 只输出改写后的问题本身：不要解释、不要引号、不要编号、不要加"改写后："之类的前缀
2. 保留原问题里的关键信息（人名、书名、术语、时间）
3. 补全指代（他/她/它/二者/这些/上述/前者…）和被省略的主语
4. 尽量沿用原句的措辞和语序，只把指代换成具体所指；
   不要换同义词、不要合并或拆分句子、不要补充对话里没出现的细节
5. 如果最新这句话本身已经完整，原样输出
6. 输出必须仍然是一个问句（以问号结尾），不要把答案写出来
"""

# 指代 / 省略信号（先剔除"其他""其中"这类非指代用法）
_ANAPHORA_WORDS = (
    "二者", "两者", "他们", "她们", "它们", "此人", "该人", "这人", "那人",
    "上述", "刚才", "前面", "以上", "这个", "那个", "这些", "那些", "此事",
    "这件事", "这次", "那次", "前者", "后者", "这里", "那里", "其中",
    "这种", "那种", "这样", "那样", "此处",
    "这本书", "该书", "这位", "那位", "该事件", "该问题", "该内容",
    "他", "她", "它", "其",
)
_NEUTRALIZE = (
    "其他", "其它", "其实", "尤其", "极其", "与其", "其余", "他人", "他日",
)

# 单个问题最多保留字符数，超长直接截断给 LLM
_MAX_HISTORY_CHARS = 400

# 改写结果的长度上限：短追问按固定上限卡（防模型写成一段解释），
# 长问题按「原问题长度 + 余量」卡，避免本身就很长的问题被误判为改写失败
_MAX_REWRITE_CHARS = 100
_REWRITE_GROWTH = 40
# 长度下限：改写只做指代补全，不该大幅缩水。
# 实测模型偶尔会越过改写直接"作答"（整句变成"意外"），这种必须判失败
_MIN_REWRITE_RATIO = 0.5


def needs_context(query: str) -> bool:
    """判断问题是否依赖上文（含指代词，或极短且无实体）"""
    text = query
    for word in _NEUTRALIZE:
        text = text.replace(word, "")

    if any(word in text for word in _ANAPHORA_WORDS):
        return True

    # 极短且没有任何专名/标点信号，通常是被省略的追问
    if len(text.strip()) <= 6 and not re.search(r"[《》「」“”\"'A-Za-z0-9]", text):
        return True

    return False


def _clean_rewrite(raw: str, original: str = "") -> Optional[str]:
    """清洗模型输出；明显不合格则返回 None 交给兜底逻辑

    长度上限随原问题放宽：改写要补全指代，结果通常比原问题长一点；
    用固定 100 字会把这类的长问题全部判失败，白白退化成拼接。
    """
    if not raw:
        return None

    text = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    text = text.strip(" \t\"'“”「」`")
    for prefix in ("改写后的问题：", "改写后：", "问题：", "问："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    limit = max(_MAX_REWRITE_CHARS, len(original.strip()) + _REWRITE_GROWTH)
    if not text or len(text) > limit:
        return None
    # 模型越过改写直接作答时，输出会缩成答案本身（如"意外"），句子结构全丢
    if len(text) < max(4, int(len(original.strip()) * _MIN_REWRITE_RATIO)):
        return None
    # 原问题是问句，改写结果也应是问句：模型自问自答时输出会变成陈述句
    if original.strip().endswith(("？", "?")) and not text.endswith(("？", "?")):
        return None
    # 模型在解释或拒答，说明改写失败
    if any(bad in text for bad in ("抱歉", "无法", "没有提供", "根据对话", "需要更多信息")):
        return None
    return text


def _history_snippet(history: list[ChatMessage]) -> list[ChatMessage]:
    """取最近两轮用于改写的上下文，assistant 内容截断以省 token"""
    tail = history[-4:]
    snippet = []
    for msg in tail:
        content = msg.content
        if not isinstance(content, str):
            continue
        if msg.role == MessageRole.assistant:
            content = content[:_MAX_HISTORY_CHARS]
        snippet.append(ChatMessage(role=msg.role, content=content))
    return snippet


def _fallback_query(history: list[ChatMessage], query: str) -> str:
    """兜底：把上一轮提问和当前追问拼起来检索（无需额外模型调用）"""
    previous = [m.content for m in history if m.role == MessageRole.user and isinstance(m.content, str)]
    if not previous:
        return query
    return f"{previous[-1].strip()} {query.strip()}"


async def rewrite_query(
    llm: Optional[LLMProvider],
    history: list[ChatMessage],
    query: str,
) -> tuple[str, str]:
    """
    返回 (用于检索的查询, 使用的方法)。

    方法取值：
    - "none"   不需要改写（首轮或已是独立问题）
    - "llm"    LLM 改写成功
    - "concat" LLM 不可用/失败，退化为拼接上一轮问题
    """
    mode = (settings.query_rewrite or "llm").lower()
    if mode == "off" or not history:
        return query, "none"

    if not needs_context(query):
        return query, "none"

    if mode == "llm" and llm is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=_REWRITE_SYSTEM),
            *_history_snippet(history),
            ChatMessage(role=MessageRole.user, content=query),
        ]
        try:
            rewritten = _clean_rewrite(await llm.chat(messages), query)
            if rewritten and rewritten != query:
                return rewritten, "llm"
            if rewritten == query:
                return query, "none"
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc)[:200])

    fallback = _fallback_query(history, query)
    if fallback != query:
        return fallback, "concat"
    return query, "none"
