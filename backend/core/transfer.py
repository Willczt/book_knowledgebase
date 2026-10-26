"""知识库 / 会话的导入导出。

导出包是一个 zip：
    manifest.json     元数据（KB 配置、文档列表、统计信息）
    vectors.ndjson    向量库快照（一行一个点，含 dense + sparse）
    sessions.ndjson   会话与消息（一行一个会话，内含其消息列表）
    files/docs/...    原始文档
    files/images/...  文档内提取的图片（按 doc_id 分目录）
    files/query_images/...  历史会话里用户上传的图片

导入时会重建 KB / 文档 / 会话 / 消息 ID 与向量点 ID，并同步改写 payload 中的
doc_id 与 image_path，因此可以反复导入同一个包而不会互相覆盖。
"""

from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db import crud
from db.models import KnowledgeBase, Session
from vector_store.qdrant_store import QdrantVectorStore

settings = get_settings()

ARCHIVE_VERSION = 2
MANIFEST_NAME = "manifest.json"
# 归档格式 v2：向量与会话改为 NDJSON（一行一个对象），可流式读写；
# 导入时仍兼容旧包里的 vectors.json 与 manifest.sessions。
VECTORS_NAME = "vectors.ndjson"
SESSIONS_NAME = "sessions.ndjson"
LEGACY_VECTORS_NAME = "vectors.json"
MAX_IMPORT_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

_PENDING_STATUSES = ("pending", "processing")
_IMPORT_BATCH_SIZE = 256  # 导入向量时的批大小


class KnowledgeBaseExists(ValueError):
    """同一实例中已存在相同 ID 的知识库，说明这个归档已经导入过了。"""


# ========================
# 小工具
# ========================

def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now().isoformat()


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def safe_filename(name: str, default: str = "archive") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name or "").strip("._")
    return cleaned or default


def _images_root() -> Path:
    return Path(settings.storage_root) / "images"


def _write_json(zf: zipfile.ZipFile, name: str, payload: Any) -> None:
    zf.writestr(
        name,
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    )


def _write_file(zf: zipfile.ZipFile, name: str, path: Path) -> None:
    zf.write(path, name, compress_type=zipfile.ZIP_STORED)


def _iter_ndjson(stream: Any, error_message: str):
    """逐行解析 NDJSON：空行跳过，格式错误抛 ValueError。

    边读边解析，不会把整个条目读进内存。
    """
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(error_message) from exc


def _read_manifest(archive: Any, expected_kind: str) -> tuple[zipfile.ZipFile, dict, set[str]]:
    """archive 可以是 bytes，也可以是已落盘的类文件对象（UploadFile.file）。"""
    if isinstance(archive, (bytes, bytearray)):
        if not archive:
            raise ValueError("导入包为空")
        if len(archive) > MAX_IMPORT_BYTES:
            raise ValueError("导入包超过 2GB，无法导入")
        source: Any = io.BytesIO(archive)
    else:
        source = archive
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass

    try:
        zf = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ValueError("导入包不是有效的 zip 文件") from exc

    # 校验失败时关闭文件句柄，成功则交给调用方 with 管理
    try:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise ValueError("导入包缺少 manifest.json")
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest.json 解析失败") from exc

        kind = manifest.get("kind")
        if kind != expected_kind:
            raise ValueError(f"导入包类型不匹配：期望 {expected_kind}，实际 {kind}")
    except Exception:
        zf.close()
        raise
    return zf, manifest, names


def _remap_image_path(
    path: Optional[str],
    old_kb_id: str,
    new_kb_id: str,
    doc_map: dict[str, str],
) -> Optional[str]:
    """把 images/{old_kb}/{old_doc}/{file} 改写成新 KB 下的路径。"""
    if not path:
        return path
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "images" and parts[1] == old_kb_id:
        new_doc = doc_map.get(parts[2])
        if new_doc:
            return "/".join(["images", new_kb_id, new_doc, *parts[3:]])
    return path


def _remap_sources(
    sources: Any,
    old_kb_id: str,
    new_kb_id: str,
    doc_map: dict[str, str],
) -> Any:
    if not isinstance(sources, list):
        return sources
    remapped = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        old_doc = item.get("doc_id")
        if old_doc:
            new_doc = doc_map.get(old_doc)
            if not new_doc:
                # 引用的文档不在包内（例如被单独删除），丢弃该引用
                continue
            item["doc_id"] = new_doc
        item["image_path"] = _remap_image_path(item.get("image_path"), old_kb_id, new_kb_id, doc_map)
        remapped.append(item)
    return remapped


async def _unique_kb_name(db: AsyncSession, base: str) -> str:
    existing = {kb.name for kb in await crud.list_kbs(db)}
    if base not in existing:
        return base
    candidate = f"{base}（导入）"
    index = 2
    while candidate in existing:
        candidate = f"{base}（导入 {index}）"
        index += 1
    return candidate


# ========================
# 导出：知识库
# ========================

async def export_kb(
    db: AsyncSession,
    kb: KnowledgeBase,
    qdrant: QdrantVectorStore,
    target: Optional[Path] = None,
) -> Any:
    """导出知识库为 zip。

    target 为文件路径时直接写入该文件（内存占用与库大小无关），返回统计信息；
    target 为 None 时返回 bytes（仅建议测试/小库使用）。
    """
    docs = await crud.list_documents(db, kb.id)
    sessions = await crud.list_sessions(db, kb.id)
    images_root = _images_root()

    manifest: dict[str, Any] = {
        "version": ARCHIVE_VERSION,
        "kind": "knowledge_base",
        "exported_at": _now(),
        "source_kb_id": kb.id,
        "knowledge_base": {
            "name": kb.name,
            "description": kb.description,
            "embedding_provider": kb.embedding_provider,
            "embedding_model_id": kb.embedding_model_id,
            "embedding_vector_dim": kb.embedding_vector_dim,
            "embedding_is_multimodal": kb.embedding_is_multimodal,
            "caption_provider": kb.caption_provider,
            "caption_model_id": kb.caption_model_id,
        },
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size_bytes": doc.file_size_bytes,
                "status": doc.status,
                "chunk_count": doc.chunk_count,
                "image_count": doc.image_count,
                "doc_metadata": doc.doc_metadata,
                "created_at": _iso(doc.created_at),
                "processing_started_at": _iso(doc.processing_started_at),
                "processing_completed_at": _iso(doc.processing_completed_at),
            }
            for doc in docs
        ],
    }

    sink = target if target is not None else io.BytesIO()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1) 向量库可能很大：从 Qdrant 边读边写，避免整库向量进内存
        vector_count = 0
        with zf.open(VECTORS_NAME, "w", force_zip64=True) as entry:
            async for point in qdrant.iter_points(kb.qdrant_collection_name):
                entry.write(json.dumps(point, ensure_ascii=False).encode("utf-8") + b"\n")
                vector_count += 1

        # 2) 会话：一次只装载一个会话的消息，内存占用与「最大的那个会话」相关，
        #    而不是随全部历史消息线性增长
        message_images: set[str] = set()
        session_count = 0
        message_count = 0
        with zf.open(SESSIONS_NAME, "w", force_zip64=True) as entry:
            for session in sessions:
                messages = await crud.get_messages(
                    db, session.id, limit=1_000_000, offset=0
                )
                payload = {
                    "id": session.id,
                    "title": session.title,
                    "llm_provider": session.llm_provider,
                    "llm_model_id": session.llm_model_id,
                    "llm_config": session.llm_config,
                    "created_at": _iso(session.created_at),
                    "updated_at": _iso(session.updated_at),
                    "messages": [
                        {
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "image_paths": msg.image_paths,
                            "sources": msg.sources,
                            "prompt_tokens": msg.prompt_tokens,
                            "completion_tokens": msg.completion_tokens,
                            "created_at": _iso(msg.created_at),
                        }
                        for msg in messages
                    ],
                }
                entry.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
                session_count += 1
                message_count += len(messages)
                for msg in messages:
                    for rel in msg.image_paths or []:
                        if isinstance(rel, str):
                            message_images.add(rel)

        # 3) manifest 放到最后写，此时统计信息才完整
        manifest["stats"] = {
            "documents": len(docs),
            "sessions": session_count,
            "messages": message_count,
            "vectors": vector_count,
        }
        _write_json(zf, MANIFEST_NAME, manifest)

        for doc in docs:
            path = Path(doc.file_path)
            if path.is_file():
                _write_file(zf, f"files/docs/{doc.id}{path.suffix}", path)

            image_dir = images_root / kb.id / doc.id
            if image_dir.is_dir():
                for image in sorted(image_dir.iterdir()):
                    if image.is_file():
                        _write_file(zf, f"files/images/{doc.id}/{image.name}", image)

        for rel in sorted(message_images):
            source = (images_root / rel).resolve()
            try:
                source.relative_to(images_root.resolve())
            except ValueError:
                continue
            if source.is_file():
                _write_file(zf, f"files/{rel}", source)

    # 写入文件时返回统计信息；传 None 时保持旧行为（返回 bytes，便于测试）
    if target is None:
        return sink.getvalue()
    return manifest["stats"]


# ========================
# 导入：知识库
# ========================

async def import_kb(
    db: AsyncSession,
    qdrant: QdrantVectorStore,
    archive: Any,
    name: Optional[str] = None,
) -> dict:
    zf, manifest, names = _read_manifest(archive, "knowledge_base")

    with zf:
        meta = manifest.get("knowledge_base") or {}
        try:
            vector_dim = int(meta.get("embedding_vector_dim") or 0)
        except (TypeError, ValueError):
            vector_dim = 0
        if vector_dim <= 0:
            raise ValueError("导入包缺少有效的 embedding 向量维度")
        if not meta.get("embedding_provider") or not meta.get("embedding_model_id"):
            raise ValueError("导入包缺少 embedding 模型信息")

        # 沿用归档里的原始 KB ID：同一实例里 ID 唯一，
        # 因此同一个归档包无法被重复导入（第二次直接判定「知识库已存在」）。
        # 跨实例导入时 ID 也保持一致，再次回导同样会被拦下。
        kb_id = str(manifest.get("source_kb_id") or "").strip() or _uid()

        existing = await crud.get_kb(db, kb_id)
        if existing is not None:
            raise KnowledgeBaseExists(
                f"知识库已存在：{existing.name}，无需重复导入"
            )

        collection_name = f"kb_{kb_id.replace('-', '')}"
        # 若上次导入中途失败留下了同名集合（DB 里没有对应记录），先清掉再重建
        await qdrant.delete_collection(collection_name)
        await qdrant.create_collection(collection_name, vector_dim)

        kb_name = await _unique_kb_name(db, name or meta.get("name") or "导入的知识库")
        try:
            kb = await crud.create_kb(db, {
                "id": kb_id,
                "name": kb_name,
                "description": meta.get("description"),
                "embedding_provider": meta["embedding_provider"],
                "embedding_model_id": meta["embedding_model_id"],
                "embedding_vector_dim": vector_dim,
                "embedding_is_multimodal": bool(meta.get("embedding_is_multimodal")),
                "caption_provider": meta.get("caption_provider"),
                "caption_model_id": meta.get("caption_model_id"),
                "qdrant_collection_name": collection_name,
            })
        except IntegrityError as exc:  # 并发导入同一归档时兜底
            await db.rollback()
            await qdrant.delete_collection(collection_name)
            raise KnowledgeBaseExists("知识库已存在，无需重复导入") from exc

        # ---------- 文档 ----------
        docs_root = Path(settings.storage_root) / "docs" / kb_id
        docs_root.mkdir(parents=True, exist_ok=True)
        images_root = _images_root()

        doc_entries = {Path(n).stem: n for n in names if n.startswith("files/docs/")}
        doc_map: dict[str, str] = {}

        for doc in manifest.get("documents") or []:
            old_id = doc.get("id")
            if not old_id:
                continue
            new_id = _uid()
            doc_map[old_id] = new_id

            entry = doc_entries.get(old_id)
            suffix = Path(entry).suffix if entry else f".{doc.get('file_type') or 'txt'}"
            dest = docs_root / f"{new_id}{suffix}"
            if entry:
                dest.write_bytes(zf.read(entry))

            image_count = 0
            prefix = f"files/images/{old_id}/"
            for name_in_zip in names:
                if not name_in_zip.startswith(prefix) or name_in_zip.endswith("/"):
                    continue
                # 只取文件名部分，防止归档里的 ../ 写到目录外
                filename = Path(name_in_zip[len(prefix):]).name
                if not filename or filename in (".", ".."):
                    continue
                target = images_root / kb_id / new_id / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name_in_zip))
                image_count += 1

            doc_status = doc.get("status") or "pending"
            if doc_status in _PENDING_STATUSES:
                # 向量随包一起导入，分块齐全的直接视为已完成
                doc_status = "indexed" if (doc.get("chunk_count") or 0) > 0 else "pending"

            error_message = None if doc_status == "indexed" else doc.get("error_message")
            if not entry:
                doc_status = "failed"
                error_message = "导入包中缺少该文档的原始文件"

            await crud.create_document(db, {
                "id": new_id,
                "kb_id": kb_id,
                "filename": doc.get("filename") or f"{new_id}{suffix}",
                "file_type": doc.get("file_type") or suffix.lstrip("."),
                "file_path": str(dest),
                "file_size_bytes": doc.get("file_size_bytes") or (dest.stat().st_size if dest.exists() else None),
                "status": doc_status,
                "error_message": error_message,
                "chunk_count": int(doc.get("chunk_count") or 0),
                "image_count": image_count or int(doc.get("image_count") or 0),
                "doc_metadata": doc.get("doc_metadata"),
                "processing_started_at": _parse_dt(doc.get("processing_started_at")),
                "processing_completed_at": _parse_dt(doc.get("processing_completed_at")),
                "created_at": _parse_dt(doc.get("created_at")) or datetime.now(),
            })

        # ---------- 向量 ----------
        # 边读边写：内存占用只跟批大小有关，与归档里的向量总量无关
        source_kb_id = manifest.get("source_kb_id") or ""
        written_vectors = 0
        pending: list[dict] = []

        def _remap_point(point: dict) -> Optional[dict]:
            payload = dict(point.get("payload") or {})
            old_doc = payload.get("doc_id")
            if old_doc and old_doc not in doc_map:
                return None
            if old_doc:
                payload["doc_id"] = doc_map[old_doc]
            if payload.get("image_path"):
                payload["image_path"] = _remap_image_path(
                    payload["image_path"], source_kb_id, kb_id, doc_map
                )
            return {
                "id": _uid(),
                "vector": point.get("vector") or {},
                "payload": payload,
            }

        async def _flush_vectors() -> None:
            nonlocal written_vectors
            if not pending:
                return
            written_vectors += await qdrant.upsert_raw(collection_name, pending)
            pending.clear()

        async def _consume_points(points: Any) -> None:
            for point in points:
                if not isinstance(point, dict):
                    continue
                remapped = _remap_point(point)
                if remapped is None:
                    continue
                pending.append(remapped)
                if len(pending) >= _IMPORT_BATCH_SIZE:
                    await _flush_vectors()

        if VECTORS_NAME in names:
            with zf.open(VECTORS_NAME) as entry:
                await _consume_points(_iter_ndjson(entry, "vectors.ndjson 解析失败"))
        elif LEGACY_VECTORS_NAME in names:
            try:
                legacy_points = json.loads(zf.read(LEGACY_VECTORS_NAME).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("vectors.json 解析失败") from exc
            await _consume_points(legacy_points or [])
        await _flush_vectors()

        # ---------- 会话与消息 ----------
        # 一次只处理一行（一个会话），内存占用与单个会话的消息数相关
        session_count = 0
        message_count = 0
        query_images_dir = images_root / "query_images"
        pending_images: list[tuple[str, Path]] = []

        async def _import_session(payload: dict) -> None:
            nonlocal session_count, message_count
            new_session_id = _uid()
            await crud.create_session(db, {
                "id": new_session_id,
                "kb_id": kb_id,
                "title": payload.get("title"),
                "llm_provider": payload.get("llm_provider") or meta.get("embedding_provider"),
                "llm_model_id": payload.get("llm_model_id") or meta.get("embedding_model_id"),
                "llm_config": payload.get("llm_config"),
                "created_at": _parse_dt(payload.get("created_at")) or datetime.now(),
                "updated_at": _parse_dt(payload.get("updated_at")) or datetime.now(),
            })
            session_count += 1

            for message in payload.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                new_message_id = _uid()
                image_paths: list[str] = []
                for rel in message.get("image_paths") or []:
                    entry_name = f"files/{rel}"
                    if entry_name not in names:
                        continue
                    suffix = Path(rel).suffix or ".png"
                    dest_rel = f"query_images/{new_message_id}{suffix}"
                    query_images_dir.mkdir(parents=True, exist_ok=True)
                    # 先登记，等 NDJSON 流读完后统一落盘：
                    # 同一个 zip 上不宜同时开着流式读取再 read 别的条目
                    pending_images.append((entry_name, images_root / dest_rel))
                    image_paths.append(dest_rel)

                await crud.create_message(db, {
                    "id": new_message_id,
                    "session_id": new_session_id,
                    "role": message.get("role") or "user",
                    "content": message.get("content") or "",
                    "image_paths": image_paths or None,
                    "sources": _remap_sources(
                        message.get("sources"), source_kb_id, kb_id, doc_map
                    ) or None,
                    "prompt_tokens": message.get("prompt_tokens"),
                    "completion_tokens": message.get("completion_tokens"),
                    "created_at": _parse_dt(message.get("created_at")) or datetime.now(),
                })
                message_count += 1

        if SESSIONS_NAME in names:
            with zf.open(SESSIONS_NAME) as entry:
                for payload in _iter_ndjson(entry, "sessions.ndjson 解析失败"):
                    if isinstance(payload, dict):
                        await _import_session(payload)
        else:
            # v1 归档：会话与消息都塞在 manifest 里
            for payload in manifest.get("sessions") or []:
                if isinstance(payload, dict):
                    await _import_session(payload)

        for entry_name, dest in pending_images:
            dest.write_bytes(zf.read(entry_name))

    return {
        "kb": kb,
        "documents": len(doc_map),
        "sessions": session_count,
        "messages": message_count,
        "vectors": written_vectors,
    }


# ========================
# 导出：会话
# ========================

async def export_session(db: AsyncSession, session: Session) -> bytes:
    messages = await crud.get_messages(db, session.id, limit=1_000_000, offset=0)
    images_root = _images_root()

    manifest = {
        "version": ARCHIVE_VERSION,
        "kind": "session",
        "exported_at": _now(),
        "session": {
            "title": session.title,
            "llm_provider": session.llm_provider,
            "llm_model_id": session.llm_model_id,
            "llm_config": session.llm_config,
            "created_at": _iso(session.created_at),
            "updated_at": _iso(session.updated_at),
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "image_paths": msg.image_paths,
                "sources": msg.sources,
                "prompt_tokens": msg.prompt_tokens,
                "completion_tokens": msg.completion_tokens,
                "created_at": _iso(msg.created_at),
            }
            for msg in messages
        ],
        "stats": {"messages": len(messages)},
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        _write_json(zf, MANIFEST_NAME, manifest)
        for msg in messages:
            for rel in msg.image_paths or []:
                if not isinstance(rel, str):
                    continue
                source = (images_root / rel).resolve()
                try:
                    source.relative_to(images_root.resolve())
                except ValueError:
                    continue
                if source.is_file():
                    _write_file(zf, f"files/{rel}", source)

    return buffer.getvalue()


# ========================
# 导入：会话
# ========================

async def import_session(db: AsyncSession, archive: bytes, kb_id: str) -> Session:
    kb = await crud.get_kb(db, kb_id)
    if kb is None:
        raise ValueError(f"目标知识库不存在：{kb_id}")

    zf, manifest, names = _read_manifest(archive, "session")

    with zf:
        meta = manifest.get("session") or {}
        session = await crud.create_session(db, {
            "id": _uid(),
            "kb_id": kb_id,
            "title": meta.get("title"),
            "llm_provider": meta.get("llm_provider") or kb.embedding_provider,
            "llm_model_id": meta.get("llm_model_id") or kb.embedding_model_id,
            "llm_config": meta.get("llm_config"),
            "created_at": _parse_dt(meta.get("created_at")) or datetime.now(),
            "updated_at": _parse_dt(meta.get("updated_at")) or datetime.now(),
        })

        available_doc_ids = {doc.id for doc in await crud.list_documents(db, kb_id)}
        images_root = _images_root()
        query_images_dir = images_root / "query_images"

        for message in manifest.get("messages") or []:
            new_message_id = _uid()
            image_paths: list[str] = []
            for rel in message.get("image_paths") or []:
                entry = f"files/{rel}"
                if entry not in names:
                    continue
                suffix = Path(rel).suffix or ".png"
                dest_rel = f"query_images/{new_message_id}{suffix}"
                query_images_dir.mkdir(parents=True, exist_ok=True)
                (images_root / dest_rel).write_bytes(zf.read(entry))
                image_paths.append(dest_rel)

            sources = []
            for item in message.get("sources") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("doc_id") and item["doc_id"] not in available_doc_ids:
                    continue
                sources.append(item)

            await crud.create_message(db, {
                "id": new_message_id,
                "session_id": session.id,
                "role": message.get("role") or "user",
                "content": message.get("content") or "",
                "image_paths": image_paths or None,
                "sources": sources or None,
                "prompt_tokens": message.get("prompt_tokens"),
                "completion_tokens": message.get("completion_tokens"),
                "created_at": _parse_dt(message.get("created_at")) or datetime.now(),
            })

    return session
