import asyncio
import uuid
from pathlib import Path
from typing import Callable, Optional

import structlog

from config import get_settings
from core.captioning.image_captioner import ImageCaptioner
from core.chunking.text_splitter import RecursiveTextSplitter
from core.embedding.base import EmbeddingProvider
from core.llm.base import LLMProvider
from core.llm.registry import get_caption_provider
from core.parsers.base import ParsedChunk
from core.parsers.epub_parser import EPUBParser
from core.parsers.markdown_parser import MarkdownParser
from core.parsers.pdf_parser import PDFParser
from core.parsers.txt_parser import TXTParser
from core.retrieval.sparse import embed_sparse
from db.models import KnowledgeBase
from vector_store.qdrant_store import ChunkPoint, QdrantVectorStore

logger = structlog.get_logger()
settings = get_settings()

# 进度回调：接收 {stage, stage_label, percent, current, total, message}
ProgressCallback = Callable[[dict], None]


class DocumentIndexer:
    """
    全文档摄入流水线：
    parse → chunk (text) + extract (images) → caption → embed (dense + sparse) → upsert
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
        kb: KnowledgeBase,
        storage_root: str,
        caption_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._kb = kb
        self._storage_root = storage_root
        self._text_splitter = RecursiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        # 知识库可指定 caption 模型，未指定时按全局优先级选择
        caption_llm = caption_provider if caption_provider is not None else get_caption_provider()
        self._captioner = ImageCaptioner(caption_llm)

    def _get_parser(self, file_path: str):
        if PDFParser.supports(file_path):
            return PDFParser()
        if EPUBParser.supports(file_path):
            return EPUBParser()
        if MarkdownParser.supports(file_path):
            return MarkdownParser()
        if TXTParser.supports(file_path):
            return TXTParser()
        raise ValueError(f"Unsupported file type: {file_path}")

    @staticmethod
    def _parse_all(parser, file_path: str, source_doc: str) -> list[ParsedChunk]:
        """同步解析（放在线程池执行，避免阻塞事件循环）"""
        chunks: list[ParsedChunk] = []
        for chunk in parser.parse(file_path):
            chunk.source_doc = source_doc
            chunks.append(chunk)
        return chunks

    def _image_storage_dir(self, doc_id: str) -> Path:
        path = Path(self._storage_root) / "images" / self._kb.id / doc_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _report(
        progress: Optional[ProgressCallback],
        stage: str,
        stage_label: str,
        percent: float,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """上报一次进度（回调异常不影响摄入）"""
        if progress is None:
            return
        try:
            progress({
                "stage": stage,
                "stage_label": stage_label,
                "percent": round(max(0.0, min(100.0, percent)), 1),
                "current": current,
                "total": total,
                "message": message,
            })
        except Exception:
            pass

    async def index_document(
        self,
        file_path: str,
        doc_id: str,
        source_doc: str,
        progress: Optional[ProgressCallback] = None,
    ) -> tuple[int, int]:
        """
        摄入单个文档。
        progress: 可选进度回调，收到 {stage, stage_label, percent, current, total, message}
        返回 (chunk_count, image_count)。
        """
        parser = self._get_parser(file_path)
        image_dir = self._image_storage_dir(doc_id)
        chunk_points: list[ChunkPoint] = []
        image_count = 0
        text_chunk_count = 0

        # ── 阶段 1：解析文档 ─────────────────────────────────────────
        logger.info("📖 [1/5] 开始解析文档", source_doc=source_doc)
        self._report(progress, "parsing", "解析文档", 1, message="正在解析文档内容…")
        parsed_chunks = await asyncio.to_thread(
            self._parse_all, parser, file_path, source_doc
        )
        logger.info("✅ 解析完成", pages_or_sections=len(parsed_chunks))
        self._report(
            progress, "parsing", "解析文档", 5,
            current=len(parsed_chunks), total=len(parsed_chunks),
            message=f"解析出 {len(parsed_chunks)} 个章节/页面",
        )

        # ── 阶段 2：分块 + 图片 Caption（并发）──────────────────────
        logger.info("✂️  [2/5] 开始分块与图片描述生成")

        # 2a: 收集 caption 任务（图片落盘推迟到确认要索引时，避免模型不可用时产生孤儿文件）
        image_tasks: list[tuple[bytes, Optional[int], str]] = []
        for parsed in parsed_chunks:
            for img in parsed.images:
                img_filename = f"{uuid.uuid4()}.{img.ext}"
                relative_path = f"images/{self._kb.id}/{doc_id}/{img_filename}"
                image_tasks.append((img.data, parsed.page_num, relative_path))

        # 2b: 并发生成所有图片描述（最多 5 个并发 API 请求）
        if image_tasks:
            concurrency = max(1, settings.caption_concurrency)
            logger.info("🖼️  并行生成图片描述", total=len(image_tasks), concurrency=concurrency)
            sem = asyncio.Semaphore(concurrency)
            total_images = len(image_tasks)
            caption_done = 0

            async def _caption_one(
                idx: int, img_data: bytes, page_num: Optional[int], relative_path: str
            ) -> Optional[tuple[str, str, Optional[int]]]:
                nonlocal caption_done
                async with sem:
                    cap = await self._captioner.caption(img_data, page_num)
                    if cap is None:
                        # 视觉模型不可用：整批跳过，不落盘、不生成占位 chunk
                        return None
                    if not cap:
                        cap = f"[图片，页面 {page_num or '?'}]"
                    # 确认要索引后再写盘
                    (image_dir / relative_path.rsplit("/", 1)[-1]).write_bytes(img_data)
                    caption_done += 1
                    logger.info(f"  ✅ 图片描述完成 {idx + 1}/{total_images}", page=page_num)
                    self._report(
                        progress, "captioning", "生成图片描述",
                        5 + 20 * caption_done / total_images,
                        current=caption_done, total=total_images,
                        message=f"图片描述 {caption_done}/{total_images}",
                    )
                    return cap, relative_path, page_num

            caption_results = await asyncio.gather(*[
                _caption_one(i, img_data, page_num, rp)
                for i, (img_data, page_num, rp) in enumerate(image_tasks)
            ])
            skipped = sum(1 for r in caption_results if r is None)
            if skipped:
                logger.warning(
                    "image_captions_skipped",
                    skipped=skipped,
                    total=len(image_tasks),
                    reason="视觉模型不可用",
                )
            caption_results = [r for r in caption_results if r is not None]

            for cap, relative_path, page_num in caption_results:
                image_count += 1
                chunk_points.append(ChunkPoint(
                    text=f"[图片描述] {cap}",
                    dense_vector=[],
                    sparse_indices=[],
                    sparse_values=[],
                    doc_id=doc_id,
                    source_doc=source_doc,
                    chunk_type="image_caption",
                    image_path=relative_path,
                    page_num=page_num,
                    chunk_index=len(chunk_points),
                ))

        # 2c: 文字分块
        for parsed in parsed_chunks:
            if parsed.text.strip():
                sub_chunks = self._text_splitter.split_text(
                    parsed.text,
                    page_num=parsed.page_num,
                    section_title=parsed.section_title,
                    chapter=parsed.chapter,
                )
                for sub in sub_chunks:
                    if not sub.text.strip():
                        continue
                    text_chunk_count += 1
                    chunk_points.append(ChunkPoint(
                        text=sub.text,
                        dense_vector=[],
                        sparse_indices=[],
                        sparse_values=[],
                        doc_id=doc_id,
                        source_doc=source_doc,
                        chunk_type="text",
                        page_num=sub.page_num,
                        chunk_index=len(chunk_points),
                        extra_metadata={
                            "section_title": sub.section_title or "",
                            "chapter": sub.chapter or "",
                        },
                    ))

        if not chunk_points:
            logger.warning("no_chunks_extracted", doc_id=doc_id, file_path=file_path)
            return 0, 0

        total_chunks = len(chunk_points)
        logger.info("✅ 分块完成", text_chunks=text_chunk_count, image_captions=image_count, total=total_chunks)
        self._report(
            progress, "chunking", "分块完成", 25,
            current=total_chunks, total=total_chunks,
            message=f"共 {total_chunks} 块（文本 {text_chunk_count} + 图片 {image_count}）",
        )

        # ── 阶段 3：Dense Embedding ──────────────────────────────────
        logger.info("🔢 [3/5] 开始向量化（dense embedding）", total_chunks=total_chunks)
        texts = [p.text for p in chunk_points]
        batch_size = self._embedding_provider.model_info.max_batch_size
        dense_vectors: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = await self._embedding_provider.embed_documents(batch)
            dense_vectors.extend(vecs)
            done = min(i + batch_size, len(texts))
            logger.info(f"  向量化进度 {done}/{len(texts)}")
            self._report(
                progress, "embedding", "向量化", 25 + 55 * done / len(texts),
                current=done, total=len(texts),
                message=f"向量化 {done}/{len(texts)} 块",
            )
        logger.info("✅ 向量化完成", vectors=len(dense_vectors))

        # ── 阶段 4：Sparse BM25 Embedding ───────────────────────────
        logger.info("🔍 [4/5] 生成稀疏向量（BM25）", total_chunks=total_chunks)
        self._report(
            progress, "sparse", "生成稀疏向量", 82,
            current=None, total=total_chunks, message="生成 BM25 稀疏向量…",
        )
        sparse_results = await asyncio.to_thread(embed_sparse, texts, 32)
        logger.info("✅ 稀疏向量完成", count=len(sparse_results))

        # 组装向量
        for i, point in enumerate(chunk_points):
            point.dense_vector = dense_vectors[i]
            point.sparse_indices, point.sparse_values = sparse_results[i]

        # ── 阶段 5：写入 Qdrant ──────────────────────────────────────
        upsert_batch = 64
        total_batches = (total_chunks + upsert_batch - 1) // upsert_batch
        logger.info("💾 [5/5] 写入向量库", total_chunks=total_chunks, batches=total_batches)
        for i in range(0, len(chunk_points), upsert_batch):
            batch = chunk_points[i : i + upsert_batch]
            await self._vector_store.upsert_points(self._kb.qdrant_collection_name, batch)
            batch_num = i // upsert_batch + 1
            logger.info(f"  写入进度 批次 {batch_num}/{total_batches}")
            done = min(i + upsert_batch, total_chunks)
            self._report(
                progress, "upserting", "写入向量库",
                88 + 12 * batch_num / total_batches,
                current=done, total=total_chunks,
                message=f"写入向量库 {batch_num}/{total_batches} 批",
            )

        self._report(
            progress, "done", "完成", 100,
            current=total_chunks, total=total_chunks,
            message=f"已入库 {total_chunks} 块",
        )
        logger.info(
            "🎉 文档入库完成",
            source_doc=source_doc,
            total_chunks=total_chunks,
            image_count=image_count,
        )
        return total_chunks, image_count
