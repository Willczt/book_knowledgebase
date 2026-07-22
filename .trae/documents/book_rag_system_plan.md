# 电子书 RAG 知识库系统 — 实现计划

## Context

从零构建一套面向电子书（epub/pdf/markdown）的生产级 RAG 系统，支持多模态输入、多模型提供商、混合检索、会话历史，以及独立的微调模块。目标：检索精度和工程质量均向生产看齐。

---

## 技术选型

| 组件 | 选型 | 原因 |
|------|------|------|
| 向量数据库 | **Qdrant** | 原生支持 dense+sparse 双向量混合检索 + RRF Fusion，单容器部署 |
| 后端框架 | **FastAPI** (async) + SQLAlchemy + Alembic | 异步原生，流式 SSE 友好 |
| 关系数据库 | **SQLite**（开发）/ **PostgreSQL**（生产 docker-compose） | 零配置开发，生产一致性 |
| PDF 解析 | **PyMuPDF (fitz)** | 最佳非扫描 PDF 文字+图片提取 |
| EPUB 解析 | **ebooklib + BeautifulSoup4** | 成熟方案，章节结构完整 |
| Markdown 解析 | **mistune** | 轻量，图片引用解析精确 |
| 图片处理 | **LLM Vision Caption**（Doubao vision / LLaVA） | 成本可控，兼容所有 embedding 模型 |
| Embedding | Doubao / Ollama / vLLM（三选一，每个 KB 锁定） | 精确管控向量空间一致性 |
| 稀疏检索 | **FastEmbed BM25** (Qdrant/bm25)，进程内 ONNX | 零外部依赖，与 Qdrant sparse vector 无缝对接 |
| Reranker | **FlashRank** ms-marco-MiniLM（可选） | 轻量 cross-encoder 二次排序 |
| 前端 | **React + Vite + TypeScript + Tailwind CSS** | 独立 dev server，分离部署灵活 |
| 微调 | **Unsloth + LoRA + SFTTrainer**（独立目录） | QLoRA 4-bit，显存友好，支持导出 GGUF 到 Ollama |

---

## 项目目录结构

```
book_knowledgebase/
├── .env.example                    # 所有环境变量模板
├── .gitignore
├── README.md                       # 快速开始 + 部署说明
├── ARCHITECTURE.md                 # Mermaid 架构图 + 组件说明
├── Makefile                        # make dev / make index / make lint
├── docker-compose.yml              # Qdrant + PostgreSQL + backend + frontend(Nginx)
├── docker-compose.dev.yml          # 开发覆盖：热重载，SQLite，无鉴权
│
├── backend/
│   ├── pyproject.toml              # uv/pip 依赖管理
│   ├── Dockerfile
│   ├── main.py                     # FastAPI app factory，lifespan startup/shutdown
│   ├── config.py                   # pydantic-settings，读 .env
│   ├── api/
│   │   ├── deps.py                 # 公共依赖：get_db, validate_kb_embedding
│   │   └── v1/
│   │       ├── router.py
│   │       ├── knowledge_base.py   # KB CRUD + /clear
│   │       ├── documents.py        # 文件上传 + 状态轮询
│   │       ├── sessions.py         # 会话 CRUD
│   │       ├── chat.py             # /chat + /chat/stream (SSE)
│   │       ├── search.py           # 裸检索（不经 LLM），调试用
│   │       └── config_routes.py    # 列出可用 embedding/LLM 模型
│   ├── core/
│   │   ├── exceptions.py           # EmbeddingModelMismatch 等自定义异常
│   │   ├── embedding/
│   │   │   ├── base.py             # EmbeddingProvider ABC + EmbeddingModelInfo
│   │   │   ├── doubao.py           # Ark API（text + vision 两端点）
│   │   │   ├── ollama.py           # POST /api/embeddings
│   │   │   ├── vllm.py             # OpenAI-compat /v1/embeddings
│   │   │   └── registry.py         # (provider,model_id) → EmbeddingProvider
│   │   ├── llm/
│   │   │   ├── base.py             # LLMProvider ABC，包含 caption_image 方法
│   │   │   ├── doubao.py           # volcenginesdkarkruntime，支持视觉
│   │   │   ├── deepseek.py         # OpenAI-compat，纯文字
│   │   │   ├── ollama.py           # llama3 / llava 双角色
│   │   │   ├── vllm.py             # OpenAI-compat
│   │   │   └── registry.py
│   │   ├── parsers/
│   │   │   ├── base.py             # DocumentParser ABC → ParsedChunk
│   │   │   ├── pdf_parser.py       # fitz：逐页文字 + 图片提取
│   │   │   ├── epub_parser.py      # ebooklib + bs4
│   │   │   └── markdown_parser.py  # mistune + 图片引用解析
│   │   ├── chunking/
│   │   │   └── text_splitter.py    # RecursiveCharacterTextSplitter
│   │   ├── captioning/
│   │   │   └── image_captioner.py  # 调用 vision LLM 生成 caption
│   │   ├── indexing/
│   │   │   └── indexer.py          # 全流水线：parse→chunk→caption→embed→upsert
│   │   ├── retrieval/
│   │   │   ├── retriever.py        # HybridRetriever: Qdrant Prefetch+RRF
│   │   │   └── reranker.py         # CrossEncoderReranker（可选）
│   │   └── rag/
│   │       └── pipeline.py         # RAGPipeline：检索→构建 prompt→SSE 流式生成
│   ├── db/
│   │   ├── engine.py               # async SQLAlchemy engine，init_db()
│   │   ├── models.py               # ORM：KnowledgeBase, Document, Session, Message
│   │   ├── schemas.py              # Pydantic v2 Request/Response schema
│   │   ├── crud.py                 # 异步 CRUD helper 函数
│   │   └── migrations/             # Alembic 迁移脚本
│   ├── vector_store/
│   │   └── qdrant_store.py         # 封装：创建集合、upsert、hybrid_search、clear
│   └── tasks/
│       └── ingest_task.py          # asyncio.create_task 后台摄入任务
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts              # 代理 /api → localhost:8000
│   ├── tailwind.config.ts
│   └── src/
│       ├── api/                    # Axios client + 各资源 API 封装
│       ├── components/
│       │   ├── layout/             # Sidebar（KB列表+会话列表）, TopBar
│       │   ├── knowledge-base/     # KBList, KBCreateModal, DocumentUploader
│       │   └── chat/               # ChatWindow, ChatInput（文字+图片）, MessageBubble, SourcePanel
│       ├── hooks/                  # useSSEChat, useKnowledgeBase
│       ├── stores/                 # Zustand：kbStore, sessionStore
│       └── types/api.ts            # 与后端 schema 对齐的 TS 类型
│
├── cli/
│   └── main.py                     # Typer CLI：kb/ingest/search/chat 命令
│
├── finetuning/                     # 独立模块，不被 backend import
│   ├── README.md
│   ├── requirements.txt            # unsloth, trl, peft, bitsandbytes
│   ├── configs/lora_config.yaml
│   ├── data/
│   │   ├── prepare_dataset.py      # 从 KB 生成 SFT 问答对
│   │   └── format_chat.py          # 应用 ChatML 模板
│   ├── train.py                    # FastLanguageModel + SFTTrainer
│   ├── evaluate.py
│   └── export_to_gguf.py           # 导出 GGUF 供 Ollama 使用
│
└── docs/
    ├── architecture.md             # Mermaid 系统图 + 组件说明
    ├── preprocessing_guide.md      # 数据预处理流程文档
    └── finetuning_guide.md         # 端到端微调指南
```

---

## 核心设计决策

### 1. Embedding 模型锁定（关键不变量）

- KB 创建时选定 `embedding_provider` + `embedding_model_id` + `embedding_vector_dim`，写入 DB 后**永不可变**
- Qdrant collection 以 `kb_{uuid_no_dashes}` 命名，创建时 `VectorParams(size=locked_dim)`
- 查询时 `deps.py` 的 `validate_kb_embedding()` 验证当前配置与 KB 锁定值匹配，不匹配立即返回 409 + 明确报错
- 稀疏向量使用 FastEmbed BM25（进程内，始终可用），不受模型锁定影响

### 2. 混合检索（Hybrid Search）

```
Qdrant prefetch[
  dense: COSINE 召回 top-20
  sparse: BM25 召回 top-20
] → RRF Fusion → top-K
→ (可选) CrossEncoder Reranker 二次排序
```

### 3. 多模态图片处理

解析文档时提取图片 → 调用 Vision LLM（Doubao vision 或 Ollama llava）生成 caption → caption 作为文本 chunk 进入索引，`payload.image_path` 指向存储文件。查询命中 caption chunk 时，前端在 SourcePanel 中展示对应图片。

### 4. 流式输出（SSE）

```
POST /sessions/{id}/chat/stream → text/event-stream
data: {"token": "..."}\n\n   × N
data: {"done": true, "sources": [...], "message_id": "..."}\n\n
```
流结束后将完整消息（含 sources）写入 DB。

### 5. 会话历史

Session → Messages (role/content/sources)。每次 RAG 请求携带最近 10 条消息作为对话历史。首条消息异步生成会话标题。

---

## 数据库 Schema（核心字段）

```
KnowledgeBase: id, name, embedding_provider*, embedding_model_id*, embedding_vector_dim*, qdrant_collection_name, created_at
Document: id, kb_id, filename, file_type, status(pending/processing/indexed/failed), chunk_count, image_count, file_path
Session: id, kb_id, title, llm_provider, llm_model_id, llm_config(JSON), updated_at
Message: id, session_id, role, content, image_paths(JSON), sources(JSON), created_at
(* = 创建后不可变)
```

---

## API 端点（v1）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/knowledge-bases` | 创建 KB（锁定 embedding 模型） |
| GET | `/knowledge-bases` | 列表 |
| DELETE | `/knowledge-bases/{id}` | 删除 KB + 向量 + 会话 |
| POST | `/knowledge-bases/{id}/clear` | **清空向量库**（保留 KB 元数据） |
| POST | `/knowledge-bases/{id}/documents` | 上传文件，异步摄入 |
| GET | `/knowledge-bases/{id}/documents/{doc_id}/status` | 状态轮询 |
| DELETE | `/knowledge-bases/{id}/documents/{doc_id}` | 删除文档及向量 |
| POST | `/sessions` | 新建会话 |
| GET | `/sessions?kb_id={id}` | 会话列表 |
| DELETE | `/sessions/{id}` | 删除会话 |
| POST | `/sessions/{id}/chat/stream` | SSE 流式问答 |
| POST | `/knowledge-bases/{id}/search` | 裸检索（调试） |
| GET | `/embedding-models` | 可用 embedding 模型列表 |
| GET | `/llm-providers` | 可用 LLM 列表 |
| GET | `/health` | DB + Qdrant 健康检查 |

---

## 实现步骤（建议顺序）

1. **项目脚手架**：目录结构、pyproject.toml、.env.example、docker-compose.yml、Makefile
2. **DB 层**：models.py → Alembic init → engine.py → crud.py → schemas.py
3. **向量存储层**：qdrant_store.py（含 create/upsert/hybrid_search/clear/delete）
4. **Embedding 层**：base.py → doubao.py → ollama.py → vllm.py → registry.py
5. **LLM 层**：base.py → doubao.py → deepseek.py → ollama.py → vllm.py → registry.py
6. **解析层**：pdf_parser.py → epub_parser.py → markdown_parser.py
7. **摄入流水线**：text_splitter.py → image_captioner.py → indexer.py → ingest_task.py
8. **检索层**：retriever.py（hybrid）→ reranker.py（可选）
9. **RAG Pipeline**：pipeline.py（prompt 构建 + SSE 流式）
10. **API 层**：所有路由 + deps.py（含 embedding 锁定校验）
11. **CLI**：typer main.py
12. **前端**：Vite 脚手架 → API client → Zustand stores → 组件（Sidebar/ChatWindow/InputArea/SourcePanel）
13. **微调模块**：独立 finetuning/ 目录 + 文档
14. **文档**：README.md、ARCHITECTURE.md（含 Mermaid 图）、preprocessing_guide.md、finetuning_guide.md

---

## 验证方案

1. 启动 Qdrant + 后端：`docker compose -f docker-compose.dev.yml up qdrant && make dev`
2. 创建 KB，上传一本 PDF，轮询状态至 `indexed`
3. 发起流式查询，验证 SSE token 逐步返回 + sources 包含正确 chunk
4. 切换错误 embedding 模型，验证 409 报错信息清晰
5. 清空 KB（`/clear`），验证 Qdrant points = 0，元数据保留
6. 前端：新建会话 → 文字提问 → 图片上传问答 → 查看会话历史
7. CLI：`python cli/main.py chat --kb-id xxx "What is backpropagation?"`
