"""知识库导入导出：流式导出、向量重映射、重复导入拦截。"""

import io
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import text

from core import transfer
from core.transfer import KnowledgeBaseExists
from db import crud


async def _seed_kb(db, storage: Path, qdrant) -> object:
    kb = await crud.create_kb(db, {
        "id": "kb-1",
        "name": "机器学习",
        "embedding_provider": "p",
        "embedding_model_id": "m",
        "embedding_vector_dim": 3,
        "qdrant_collection_name": "kb_kb1",
    })
    await qdrant.create_collection("kb_kb1", 3)

    docs_dir = storage / "docs" / "kb-1"
    docs_dir.mkdir(parents=True)
    (docs_dir / "d1.md").write_bytes(b"# hello")
    images_dir = storage / "images" / "kb-1" / "d1"
    images_dir.mkdir(parents=True)
    (images_dir / "p1.png").write_bytes(b"PNG")

    await crud.create_document(db, {
        "id": "d1",
        "kb_id": "kb-1",
        "filename": "a.md",
        "file_type": "md",
        "file_path": str(docs_dir / "d1.md"),
        "file_size_bytes": 7,
        "status": "indexed",
        "chunk_count": 1,
        "image_count": 1,
    })
    qdrant.collections["kb_kb1"] = [{
        "id": "p0",
        "vector": {"dense": [0.1, 0.2, 0.3], "sparse": {"indices": [1], "values": [0.5]}},
        "payload": {
            "text": "hello",
            "doc_id": "d1",
            "source_doc": "a.md",
            "chunk_type": "image_caption",
            "image_path": "images/kb-1/d1/p1.png",
            "chunk_index": 0,
        },
    }]
    return kb


async def _clear_tables(db) -> None:
    for table in ("messages", "sessions", "documents", "knowledge_bases"):
        await db.execute(text(f"DELETE FROM {table}"))
    await db.commit()
    # 清掉 identity map，否则重新插入同 id 会触发 SAWarning
    db.expunge_all()


async def test_export_writes_zip_and_streams_vectors(db, qdrant, storage):
    kb = await _seed_kb(db, storage, qdrant)

    out = storage / "export.zip"
    stats = await transfer.export_kb(db, kb, qdrant, out)

    assert stats["vectors"] == 1 and stats["documents"] == 1
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        vectors = [
            json.loads(line) for line in zf.read("vectors.ndjson").splitlines() if line.strip()
        ]

    assert "files/docs/d1.md" in names
    assert "sessions.ndjson" in names
    assert manifest["stats"]["vectors"] == 1
    assert manifest["stats"]["sessions"] == 0
    assert "sessions" not in manifest
    assert vectors[0]["id"] == "p0"
    assert manifest["source_kb_id"] == "kb-1"


async def test_import_remaps_ids_and_blocks_duplicates(db, qdrant, storage):
    kb = await _seed_kb(db, storage, qdrant)
    out = storage / "export.zip"
    await transfer.export_kb(db, kb, qdrant, out)

    # 源库还在时不允许重复导入
    with pytest.raises(KnowledgeBaseExists):
        with out.open("rb") as fh:
            await transfer.import_kb(db, qdrant, fh)

    await _clear_tables(db)

    # 用文件对象导入（模拟 UploadFile.file，不整包读内存）
    with out.open("rb") as fh:
        result = await transfer.import_kb(db, qdrant, fh)

    assert result["documents"] == 1 and result["vectors"] == 1
    new_kb = result["kb"]
    assert new_kb.id == "kb-1"

    docs = await crud.list_documents(db, new_kb.id)
    assert len(docs) == 1 and Path(docs[0].file_path).is_file()

    point = qdrant.collections[new_kb.qdrant_collection_name][0]
    assert point["payload"]["doc_id"] == docs[0].id
    assert point["payload"]["image_path"] == f"images/{new_kb.id}/{docs[0].id}/p1.png"


async def test_export_import_roundtrip_sessions(db, qdrant, storage):
    """会话与消息走 NDJSON 流；带图片的消息往返后图片仍然可用。"""
    kb = await _seed_kb(db, storage, qdrant)
    await crud.create_session(db, {
        "id": "s1",
        "kb_id": kb.id,
        "title": "历史会话",
        "llm_provider": "p",
        "llm_model_id": "m",
    })
    image = storage / "images" / "query_images" / "u1.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"PNG")
    await crud.create_message(db, {
        "id": "m1",
        "session_id": "s1",
        "role": "user",
        "content": "hi",
        "image_paths": ["query_images/u1.png"],
    })

    out = storage / "export.zip"
    stats = await transfer.export_kb(db, kb, qdrant, out)
    assert stats["sessions"] == 1 and stats["messages"] == 1

    await _clear_tables(db)
    with out.open("rb") as fh:
        result = await transfer.import_kb(db, qdrant, fh)

    assert result["sessions"] == 1 and result["messages"] == 1
    sessions = await crud.list_sessions(db, result["kb"].id)
    messages = await crud.get_messages(db, sessions[0].id, limit=10, offset=0)
    assert len(messages) == 1 and messages[0].content == "hi"
    rel = messages[0].image_paths[0]
    assert (storage / "images" / rel).is_file()


async def test_invalid_archive_rejected(db, qdrant, storage):
    with pytest.raises(ValueError):
        await transfer.import_kb(db, qdrant, b"not-a-zip")


async def test_import_legacy_archive(db, qdrant, storage):
    """v1 归档（vectors.json + manifest.sessions）仍然可以导入。"""
    legacy_manifest = {
        "version": 1,
        "kind": "knowledge_base",
        "source_kb_id": "kb-legacy",
        "knowledge_base": {
            "name": "旧包",
            "embedding_provider": "p",
            "embedding_model_id": "m",
            "embedding_vector_dim": 3,
        },
        "documents": [],
        "sessions": [
            {
                "id": "s1",
                "title": "历史会话",
                "messages": [{"id": "m1", "role": "user", "content": "hi"}],
            }
        ],
    }
    legacy_vectors = [
        {"id": "v1", "vector": {"dense": [0.1, 0.2, 0.3]}, "payload": {"text": "hi"}}
    ]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(legacy_manifest))
        zf.writestr("vectors.json", json.dumps(legacy_vectors))

    buffer.seek(0)
    result = await transfer.import_kb(db, qdrant, buffer)

    assert result["vectors"] == 1
    assert result["sessions"] == 1 and result["messages"] == 1
    assert len(qdrant.collections[result["kb"].qdrant_collection_name]) == 1
