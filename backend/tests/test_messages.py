"""会话消息读取：最近 N 条 / 向前翻页 / 时间精度。"""

from db import crud


async def _seed(db, count: int = 120) -> None:
    await crud.create_kb(db, {
        "id": "kb-1",
        "name": "测试库",
        "embedding_provider": "p",
        "embedding_model_id": "m",
        "embedding_vector_dim": 3,
        "qdrant_collection_name": "kb_kb1",
    })
    await crud.create_session(db, {
        "id": "s-1",
        "kb_id": "kb-1",
        "llm_provider": "p",
        "llm_model_id": "m",
        "title": "t",
    })
    for i in range(count):
        await crud.create_message(db, {
            "id": f"m{i}",
            "session_id": "s-1",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"msg{i}",
        })


async def test_latest_returns_newest_page(db):
    await _seed(db)

    latest = await crud.get_messages(db, "s-1", limit=50, latest=True)

    assert len(latest) == 50
    assert latest[-1].content == "msg119"
    # 返回时必须是升序，前端可以直接渲染
    assert [m.created_at for m in latest] == sorted(m.created_at for m in latest)


async def test_before_paging_has_no_overlap(db):
    await _seed(db)

    latest = await crud.get_messages(db, "s-1", limit=50, latest=True)
    older = await crud.get_messages(db, "s-1", limit=50, before=latest[0].created_at)

    assert len(older) == 50
    assert older[-1].content == "msg69"
    # 两页严格不重叠
    assert older[-1].created_at < latest[0].created_at


async def test_before_excludes_second_precision_boundary(db):
    """老数据时间戳只到秒，翻页游标不能把同一条消息再取一次"""
    from datetime import datetime

    await crud.create_kb(db, {
        "id": "kb-1", "name": "测试库", "embedding_provider": "p",
        "embedding_model_id": "m", "embedding_vector_dim": 3,
        "qdrant_collection_name": "kb_kb1",
    })
    await crud.create_session(db, {
        "id": "s-1", "kb_id": "kb-1", "llm_provider": "p",
        "llm_model_id": "m", "title": "t",
    })
    for i in range(2):
        await crud.create_message(db, {
            "id": f"m{i}", "session_id": "s-1", "role": "user",
            "content": f"msg{i}",
            "created_at": datetime(2026, 1, 1, 11, 59, i),
        })
    await crud.create_message(db, {
        "id": "legacy",
        "session_id": "s-1",
        "role": "user",
        "content": "legacy",
        "created_at": datetime(2026, 1, 1, 12, 0, 0),  # 无微秒，模拟旧数据
    })

    page = await crud.get_messages(db, "s-1", limit=5, before=datetime(2026, 1, 1, 12, 0, 0))

    assert [m.id for m in page] == ["m0", "m1"]


async def test_messages_have_microsecond_ordering(db):
    await _seed(db, count=4)

    messages = await crud.get_messages(db, "s-1", limit=10)

    assert [m.content for m in messages] == ["msg0", "msg1", "msg2", "msg3"]
    # 逐条写入的时间戳必须严格递增（秒级精度会退化成相等）
    stamps = [m.created_at for m in messages]
    assert len(set(stamps)) == len(stamps)
