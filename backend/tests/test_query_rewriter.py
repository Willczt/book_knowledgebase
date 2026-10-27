"""多轮对话的检索查询改写：指代判定、输出清洗、失败兜底。"""

import pytest

from core.llm.base import ChatMessage, MessageRole
from core.rag import query_rewriter as qr


class _FakeLLM:
    """只实现 chat()，用于验证改写调用与降级路径"""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.reply


def _history(*pairs: tuple[str, str]) -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole(role), content=content) for role, content in pairs
    ]


_FOLLOW_UP = "二者矛盾体现在哪里？"
_STANDALONE = "曾国藩和李鸿章的矛盾体现在哪里？"


@pytest.mark.parametrize(
    "query",
    [
        "二者矛盾体现在哪里？",
        "他的主要贡献是什么？",
        "上述观点是否成立？",
        "这本书的作者是谁",
        "这种情况下应该先回应哪一种情绪？",
        "为什么",
    ],
)
def test_needs_context_detects_follow_up(query: str) -> None:
    assert qr.needs_context(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "曾国藩和李鸿章的矛盾体现在哪里？",
        "《曾国藩家书》讲了什么",
        "尤其要注意哪些修身方法",  # 「尤其」里的「其」不是指代
        "其余的章节讲了什么",  # 「其余」同理
        "What did Zeng Guofan write?",
    ],
)
def test_needs_context_accepts_standalone(query: str) -> None:
    assert qr.needs_context(query) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("改写后的问题：曾国藩和李鸿章的矛盾体现在哪里？", _STANDALONE),
        ('"曾国藩和李鸿章的矛盾体现在哪里？"', _STANDALONE),
        ("`曾国藩和李鸿章的矛盾体现在哪里？`", _STANDALONE),
        ("问：二者有何分歧\n补充说明：……", "二者有何分歧"),
        ("", None),
        ("   ", None),
        ("抱歉，我无法根据对话完成改写。", None),
        ("需要更多信息才能改写这个问题。", None),
        ("改写后：" + "很" * 120, None),
    ],
)
def test_clean_rewrite(raw: str, expected: str | None) -> None:
    assert qr._clean_rewrite(raw) == expected


_LONG_ORIGINAL = "在汉朝那次跨海进攻朝鲜的战役里，" + "关于指挥关系和作战安排上的具体问题，" * 6 + "还有哪些明显问题？"
_LONG_REWRITTEN = "汉武帝派兵跨海进攻朝鲜的战役里，" + "关于指挥关系和作战安排上的具体问题，" * 6 + "还有哪些明显问题？"


def test_clean_rewrite_allows_long_question() -> None:
    """长问题补全指代后本就超过 100 字，不应被当成改写失败"""
    assert len(_LONG_REWRITTEN) > qr._MAX_REWRITE_CHARS
    assert qr._clean_rewrite(_LONG_REWRITTEN, _LONG_ORIGINAL) == _LONG_REWRITTEN


def test_clean_rewrite_rejects_rambling_even_for_long_question() -> None:
    runaway = _LONG_REWRITTEN + "改写说明：" + "补" * 200

    assert qr._clean_rewrite(runaway, _LONG_ORIGINAL) is None


def test_clean_rewrite_rejects_answer_instead_of_rewrite() -> None:
    """实测模型会把整句改写成答案本身（"意外"），这种必须退回兜底"""
    question = "这个框架里排在“结果”后面、用来引入超出努力范畴的变数的环节是什么？"

    assert qr._clean_rewrite("意外", question) is None


def test_clean_rewrite_rejects_statement_for_question() -> None:
    """模型自问自答时会输出陈述句，问句改写必须是问句"""
    question = "他为什么认为真正有效的学习不能只靠看书，而必须自己动笔写、并用自己的话讲出来？"
    statement = (
        "在《认知觉醒》里，周岭并不是否定看书，而是否定“只靠看书”这种被动输入。"
        "他认为真正有效的学习必须动笔写、用自己的话讲出来。"
    )

    assert qr._clean_rewrite(statement, question) is None


async def test_mode_off_skips_rewrite(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "off")
    llm = _FakeLLM(reply=_STANDALONE)
    history = _history(("user", "曾国藩和李鸿章有何分歧？"), ("assistant", "……"))

    assert await qr.rewrite_query(llm, history, _FOLLOW_UP) == (_FOLLOW_UP, "none")
    assert llm.calls == []


async def test_no_history_skips_rewrite(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply=_STANDALONE)

    assert await qr.rewrite_query(llm, [], _FOLLOW_UP) == (_FOLLOW_UP, "none")
    assert llm.calls == []


async def test_standalone_query_skips_llm(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply="不该被调用")
    history = _history(("user", "介绍一下晚清名臣"), ("assistant", "……"))

    assert await qr.rewrite_query(llm, history, _STANDALONE) == (_STANDALONE, "none")
    assert llm.calls == []


async def test_llm_rewrite_success(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply=_STANDALONE)
    history = _history(("user", "曾国藩和李鸿章有何分歧？"), ("assistant", "……"))

    query, method = await qr.rewrite_query(llm, history, _FOLLOW_UP)

    assert (query, method) == (_STANDALONE, "llm")
    sent = llm.calls[0]
    assert sent[0].role == MessageRole.system
    assert sent[-1] == ChatMessage(role=MessageRole.user, content=_FOLLOW_UP)
    assert [m.content for m in sent[1:-1]] == ["曾国藩和李鸿章有何分歧？", "……"]


async def test_llm_error_falls_back_to_concat(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(error=RuntimeError("timeout"))
    history = _history(("user", "曾国藩和李鸿章有何分歧？"), ("assistant", "……"))

    query, method = await qr.rewrite_query(llm, history, _FOLLOW_UP)

    assert method == "concat"
    assert query == f"曾国藩和李鸿章有何分歧？ {_FOLLOW_UP}"


async def test_unusable_reply_falls_back_to_concat(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply="抱歉，我无法根据对话改写这个问题。")
    history = _history(("user", "曾国藩和李鸿章有何分歧？"), ("assistant", "……"))

    query, method = await qr.rewrite_query(llm, history, _FOLLOW_UP)

    assert method == "concat"
    assert query.startswith("曾国藩和李鸿章有何分歧？")


async def test_unchanged_reply_needs_no_rewrite(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply=_FOLLOW_UP)
    history = _history(("user", "曾国藩和李鸿章有何分歧？"), ("assistant", "……"))

    assert await qr.rewrite_query(llm, history, _FOLLOW_UP) == (_FOLLOW_UP, "none")


async def test_concat_mode_never_calls_llm(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "concat")
    llm = _FakeLLM(reply=_STANDALONE)
    history = _history(("user", "曾国藩和李鸿章有何分歧？"), ("assistant", "……"))

    query, method = await qr.rewrite_query(llm, history, _FOLLOW_UP)

    assert method == "concat"
    assert query == f"曾国藩和李鸿章有何分歧？ {_FOLLOW_UP}"
    assert llm.calls == []


async def test_history_snippet_truncates_assistant_content(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply=_STANDALONE)
    history = _history(
        ("user", "第一轮问题"),
        ("assistant", "旧" * 1000),
        ("user", "第二轮问题"),
        ("assistant", "新" * 1000),
    )

    await qr.rewrite_query(llm, history, _FOLLOW_UP)

    sent = llm.calls[0][1:-1]
    assert len(sent) == 4
    assert sent[0].content == "第一轮问题"  # 只截断 assistant
    for message in (sent[1], sent[3]):
        assert message.role == MessageRole.assistant
        assert len(message.content) == qr._MAX_HISTORY_CHARS


async def test_only_last_two_turns_are_used(monkeypatch) -> None:
    monkeypatch.setattr(qr.settings, "query_rewrite", "llm")
    llm = _FakeLLM(reply=_STANDALONE)
    history = _history(
        ("user", "很早以前的问题"),
        ("assistant", "很早以前的回答"),
        ("user", "曾国藩和李鸿章有何分歧？"),
        ("assistant", "……"),
        ("user", "二者谁的影响更大？"),
        ("assistant", "……"),
    )

    await qr.rewrite_query(llm, history, _FOLLOW_UP)

    sent = llm.calls[0][1:-1]
    assert [m.content for m in sent] == [
        "曾国藩和李鸿章有何分歧？",
        "……",
        "二者谁的影响更大？",
        "……",
    ]
