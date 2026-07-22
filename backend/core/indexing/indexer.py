import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional

import structlog

from config import get_settings
from core.captioning.image_captioner import ImageCaptioner
from core.chunking.text_splitter import RecursiveTextSplitter
from core.embedding.base import EmbeddingProvider
from core.embedding.registry import get_embedding_registry
from core.llm.registry import get_caption_provider
from core.parsers.base import ParsedChunk
from core.parsers.epub_parser import EPUBParser
from core.parsers.markdown_parser import MarkdownParser
from core.parsers.pdf_parser import PDFParser
from vector_store.qdrant_store import ChunkPoint, QdrantVectorStore
from db.models import KnowledgeBase

logger = structlog.get_logger()
settings = get_settings()


def _get_sparse_encoder():
    """延迟初始化 FastEmbed BM25 稀疏编码器"""
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")


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
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._kb = kb
        self._storage_root = storage_root
        self._text_splitter = RecursiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        caption_llm = get_caption_provider()
        self._captioner = ImageCaptioner(caption_llm)
        self._sparse_encoder = _get_sparse_encoder()

    def _get_parser(self, file_path: str):
        if PDFParser.supports(file_path):
            return PDFParser()
        if EPUBParser.supports(file_path):
            return EPUBParser()
        if MarkdownParser.supports(file_path):
            return MarkdownParser()
        raise ValueError(f"Unsupported file type: {file_path}")

    def _image_storage_dir(self, doc_id: str) -> Path:
        path = Path(self._storage_root) / "images" / self._kb.id / doc_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def index_document(
        self,
        file_path: str,
        doc_id: str,
        source_doc: str,
    ) -> tuple[int, int]:
        """
        摄入单个文档。
        返回 (chunk_count, image_count)。
        """
        parser = self._get_parser(file_path)
        image_dir = self._image_storage_dir(doc_id)
        chunk_points: list[ChunkPoint] = []
        image_count = 0
        text_chunk_count = 0

        # ── 阶段 1：解析文档 ─────────────────────────────────────────
        logger.info("📖 [1/5] 开始解析文档", source_doc=source_doc)
        parsed_chunks: list[ParsedChunk] = []
        for chunk in parser.parse(file_path):
            chunk.source_doc = source_doc
            parsed_chunks.append(chunk)
        logger.info("✅ 解析完成", pages_or_sections=len(parsed_chunks))

        # ── 阶段 2：分块 + 图片 Caption ─────────────────────────────
        logger.info("✂️  [2/5] 开始分块与图片描述生成")
        for parsed in parsed_chunks:
            # 处理图片
            for img in parsed.images:
                img_filename = f"{uuid.uuid4()}.{img.ext}"
                img_path = image_dir / img_filename
                img_path.write_bytes(img.data)
                relative_path = f"images/{self._kb.id}/{doc_id}/{img_filename}"

                image_count += 1
                logger.info("🖼️  生成图片描述", image_index=image_count, page=parsed.page_num)
                caption = await self._captioner.caption(img.data, parsed.page_num)
                if not caption:
                    caption = f"[图片，页面 {parsed.page_num or '?'}]"

                chunk_points.append(ChunkPoint(
                    text=f"[图片描述] {caption}",
                    dense_vector=[],
                    sparse_indices=[],
                    sparse_values=[],
                    doc_id=doc_id,
                    source_doc=source_doc,
                    chunk_type="image_caption",
                    image_path=relative_path,
                    page_num=parsed.page_num,
                    chunk_index=len(chunk_points),
                ))

            # 处理文字
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
        logger.info("✅ 向量化完成", vectors=len(dense_vectors))

        # ── 阶段 4：Sparse BM25 Embedding ───────────────────────────
        logger.info("🔍 [4/5] 生成稀疏向量（BM25）", total_chunks=total_chunks)
        sparse_results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(self._sparse_encoder.embed(texts, batch_size=32)),
        )
        logger.info("✅ 稀疏向量完成", count=len(sparse_results))

        # 组装向量
        for i, point in enumerate(chunk_points):
            point.dense_vector = dense_vectors[i]
            sparse = sparse_results[i]
            point.sparse_indices = sparse.indices.tolist()
            point.sparse_values = sparse.values.tolist()

        # ── 阶段 5：写入 Qdrant ──────────────────────────────────────
        upsert_batch = 64
        total_batches = (total_chunks + upsert_batch - 1) // upsert_batch
        logger.info("💾 [5/5] 写入向量库", total_chunks=total_chunks, batches=total_batches)
        for i in range(0, len(chunk_points), upsert_batch):
            batch = chunk_points[i : i + upsert_batch]
            await self._vector_store.upsert_points(self._kb.qdrant_collection_name, batch)
            batch_num = i // upsert_batch + 1
            logger.info(f"  写入进度 批次 {batch_num}/{total_batches}")

        logger.info(
            "🎉 文档入库完成",
            source_doc=source_doc,
            total_chunks=total_chunks,
            image_count=image_count,
        )
        return total_chunks, image_count
