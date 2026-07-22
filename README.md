# Book Knowledgebase RAG System

> 生产级电子书 RAG 知识库系统 — 支持多模态、多模型、混合检索、流式问答

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-purple)](https://qdrant.tech)

---

## 功能特性

- 📚 **多格式支持**：PDF（图文混排）、EPUB、Markdown
- 🖼️ **多模态**：图片自动通过 Vision LLM 生成描述并索引
- 🔍 **混合检索**：Dense（向量）+ Sparse（BM25）双路召回，RRF 融合 + CrossEncoder 重排
- 🔒 **Embedding 锁定**：防止向量空间错位的静默失败
- 💬 **流式问答**：SSE 实时输出，历史会话管理
- 🌐 **多模型**：豆包（doubao-seed）、DeepSeek、Ollama（本地）、vLLM
- 🖥️ **双端入口**：React 前端（豆包风格）+ CLI 终端工具
- 🔧 **微调模块**：QLoRA 垂直领域微调，导出 GGUF 到 Ollama

---

## 系统架构

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（含 Mermaid 架构图）。

```
前端 (React)  ─┐
CLI (Typer)   ─┤─→  FastAPI  ─→  RAG Pipeline  ─→  Qdrant (向量) + SQLite (元数据)
               └           ↑
                      LLM / Embedding
                  (豆包 / DeepSeek / Ollama / vLLM)
```

---

## 快速开始

### 前置要求

- Docker + Docker Compose（用于 Qdrant）
- Python 3.11+
- Node.js 20+

### 1. 克隆并配置环境变量

```bash
git clone <repo>
cd book_knowledgebase
cp .env.example .env
```

编辑 `.env`，必填项：
```env
# 豆包（火山引擎 Ark）
ARK_API_KEY=your_key
DOUBAO_CHAT_MODEL=doubao-seed-1-6-250715
DOUBAO_EMBEDDING_MODEL=doubao-embedding-text-240715

# 或 DeepSeek
DEEPSEEK_API_KEY=your_key
```

### 2. 启动服务

```bash
# 启动 Qdrant（向量库）
make up

# 后端（另一个终端）
make dev-backend

# 前端（另一个终端）
make dev-frontend
```

打开 http://localhost:5173 访问前端。

### 3. 导入电子书

**通过前端**：
1. 点击「新建知识库」，选择 Embedding 模型
2. 展开知识库，拖拽上传 PDF/EPUB/MD 文件
3. 等待绿色状态点（索引完成）
4. 点击「新对话」，选择 LLM，开始问答

**通过 CLI**：
```bash
pip install -e backend/
book-rag kb create "我的书库" --provider doubao
book-rag ingest --kb-id <id> path/to/book.pdf
book-rag chat --kb-id <id>
```

---

## 项目结构

```
book_knowledgebase/
├── backend/                  # FastAPI 后端
│   ├── api/v1/               # 路由：知识库、文档、会话、检索、配置
│   ├── core/
│   │   ├── embedding/        # Embedding 多提供商（豆包、Ollama、vLLM）
│   │   ├── llm/              # LLM 多提供商（豆包、DeepSeek、Ollama、vLLM）
│   │   ├── parsers/          # 文档解析（PDF、EPUB、Markdown）
│   │   ├── chunking/         # 递归文本分块
│   │   ├── captioning/       # 图片 Vision Caption
│   │   ├── indexing/         # 摄入流水线
│   │   ├── retrieval/        # 混合检索 + Reranker
│   │   └── rag/              # RAG Pipeline（流式）
│   ├── db/                   # ORM 模型、CRUD、Schema、Alembic 迁移
│   ├── tasks/                # 后台摄入任务
│   ├── vector_store/         # Qdrant 向量存储
│   ├── config.py             # pydantic-settings 配置
│   └── main.py               # FastAPI App 入口
├── frontend/                 # React 前端
│   ├── src/
│   │   ├── api/              # API 客户端（axios）
│   │   ├── components/
│   │   │   ├── chat/         # ChatWindow、ChatInput、MessageBubble
│   │   │   └── layout/       # Sidebar（KB + 会话管理）
│   │   ├── hooks/            # useSSEChat（SSE 流式）
│   │   ├── stores/           # Zustand 状态管理
│   │   └── types/            # TypeScript 类型定义
│   └── Dockerfile            # Nginx 生产构建
├── cli/main.py               # Typer CLI 工具
├── finetuning/               # 可选：垂直场景微调模块
│   ├── data/                 # 数据准备脚本
│   ├── train.py              # Unsloth LoRA 训练
│   ├── export_to_gguf.py     # 导出到 Ollama
│   └── evaluate.py           # 效果评估
├── docs/
│   ├── ARCHITECTURE.md       # 系统架构图（Mermaid）
│   ├── preprocessing_guide.md # 数据预处理指南
│   └── finetuning_guide.md   # 微调指南
├── docker-compose.yml        # 生产环境
├── docker-compose.dev.yml    # 开发覆盖（仅 Qdrant）
├── Makefile                  # 常用命令
└── .env.example              # 环境变量模板
```

---

## 关键设计决策

### Embedding 模型锁定

知识库创建时绑定 `(provider, model_id, vector_dim)` 三元组。后续所有摄入和检索操作都会校验该三元组，不匹配返回 `409 Conflict`。

**原因**：不同 Embedding 模型生成的向量空间不同，若摄入和检索使用了不同模型，余弦相似度计算失去意义，会导致静默的检索失败（不报错但结果完全错误）。

### 混合检索策略

- **Dense**: Embedding 模型生成的语义向量（float32）
- **Sparse**: FastEmbed BM25 (`Qdrant/bm25`)，进程内 ONNX，关键词精确匹配
- **融合**: Qdrant 原生 Prefetch + RRF（倒数排名融合），无需手动合并列表
- **重排**: FlashRank CrossEncoder ms-marco-MiniLM，可选第二阶段精排

### 图片处理

图片不直接做向量检索（vision embedding 成本高），而是：
1. Vision LLM 生成中文描述
2. 描述作为普通文字 chunk 索引
3. Payload 记录 `image_path`，命中时前端展示原图

---

## CLI 使用

```bash
# 知识库管理
book-rag kb list
book-rag kb create "Deep Learning" --provider doubao
book-rag kb clear <kb-id>
book-rag kb delete <kb-id>

# 摄入文档
book-rag ingest --kb-id <id> book.pdf
book-rag ingest --kb-id <id> books/  # 整个目录

# 检索
book-rag search --kb-id <id> "什么是反向传播？" --top-k 5

# 交互式聊天
book-rag chat --kb-id <id> --provider doubao --model doubao-seed-1-6-250715
```

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/db.sqlite3` | 数据库 URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 地址 |
| `ARK_API_KEY` | - | 豆包 API Key |
| `DOUBAO_CHAT_MODEL` | `doubao-seed-1-6-250715` | 豆包 Chat 模型 |
| `DOUBAO_EMBEDDING_MODEL` | `doubao-embedding-text-240715` | 豆包 Embedding 模型 |
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `VLLM_BASE_URL` | `http://localhost:8001` | vLLM 地址 |
| `CHUNK_SIZE` | `1000` | 分块大小 |
| `CHUNK_OVERLAP` | `200` | 分块重叠 |
| `ENABLE_RERANKER` | `true` | 是否启用 CrossEncoder 重排 |
| `TOP_K_RETRIEVAL` | `20` | 检索召回数 |
| `TOP_K_RERANK` | `5` | 重排后保留数 |
| `UPLOAD_DIR` | `./data/uploads` | 文件上传目录 |
| `IMAGE_CACHE_DIR` | `./data/images` | 图片缓存目录 |

---

## 文档

- [系统架构图](docs/ARCHITECTURE.md)
- [数据预处理指南](docs/preprocessing_guide.md)
- [微调指南](docs/finetuning_guide.md)
- [微调模块说明](finetuning/README.md)

---

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + asyncio |
| 向量数据库 | Qdrant（dense + sparse 双向量） |
| 关系数据库 | SQLite（开发）/ PostgreSQL（生产） |
| ORM | SQLAlchemy Async + Alembic |
| 前端 | React 18 + Vite + TypeScript + Tailwind CSS |
| 状态管理 | Zustand + TanStack Query |
| Embedding | 豆包 / Ollama / vLLM + FastEmbed BM25 |
| LLM | 豆包 / DeepSeek / Ollama / vLLM |
| 重排 | FlashRank CrossEncoder |
| 微调 | Unsloth + LoRA + SFTTrainer |
| 日志 | structlog |
| CLI | Typer + Rich |
