# Book Knowledgebase RAG 系统使用指南

> **已过时**：本文档为 v1.0 历史版本，其中 `ARK_API_KEY` 为必填的说明已不适用（豆包/Ark 现为可选 provider）。最新文档请见 [USAGE.md](./USAGE.md)。

> 版本：v1.0 | 更新日期：2026-07-22 | 适用环境：Python 3.12 / Node.js 20+

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境准备](#2-环境准备)
3. [配置说明](#3-配置说明)
4. [快速开始（5 分钟上手）](#4-快速开始5-分钟上手)
5. [前端界面使用指南](#5-前端界面使用指南)
6. [CLI 使用指南](#6-cli-使用指南)
7. [知识库管理](#7-知识库管理)
8. [对话功能使用](#8-对话功能使用)
9. [数据预处理原理](#9-数据预处理原理)
10. [模型配置指南](#10-模型配置指南)
11. [常见问题排查](#11-常见问题排查)
12. [高级配置](#12-高级配置)

---

## 1. 系统概述

Book Knowledgebase 是一套面向电子书的**生产级 RAG（检索增强生成）系统**，支持将 PDF、EPUB、Markdown 格式的书籍构建成可问答的智能知识库。

### 核心能力

| 能力 | 说明 |
|------|------|
| **多格式解析** | PDF（含图文混排）、EPUB、Markdown，图片自动 Caption |
| **混合检索** | Dense（语义向量）+ Sparse（BM25 关键词）双路召回，Qdrant RRF 融合 |
| **多模型支持** | 豆包（doubao）、DeepSeek、Ollama（本地）、vLLM，可热切换 |
| **Embedding 锁定** | 知识库创建时绑定 Embedding 模型，防止向量空间错位的静默失败 |
| **SSE 流式回答** | 实时 token 输出，前端无缝渲染 |
| **多会话管理** | 独立会话历史，最多保留 10 轮上下文 |
| **双入口** | React 18 前端（Web UI）+ Typer CLI（终端工具） |

### 技术架构总览

```
┌─────────────────┬──────────────────────┐
│  React 前端      │     CLI (Typer)      │
│  (Vite + TS +   │     bookrag kb/      │
│   Tailwind)     │     ingest/chat      │
└────────┬────────┴──────────┬───────────┘
         │                   │
         ▼                   ▼
┌─────────────────────────────────────────┐
│          FastAPI (port 8000)            │
│       REST API + SSE Streaming          │
│         /api/v1/*                       │
└──────────┬──────────────┬──────────────┘
           │              │
    ┌──────▼──────┐  ┌────▼──────────────┐
    │ RAG Pipeline│  │  摄入流水线        │
    │ 检索→Prompt │  │ 解析→分块→Caption │
    │ →LLM 流式   │  │ →向量化→入库      │
    └──────┬──────┘  └────┬──────────────┘
           │               │
    ┌──────▼───────────────▼──────────────┐
    │           存储层                     │
    │  Qdrant (向量)  SQLite (元数据)      │
    │  Dense + Sparse  KnowledgeBase/      │
    │  双向量索引      Document/Session    │
    └─────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │           模型层                     │
    │  Embedding: 豆包 / Ollama / vLLM    │
    │  LLM: 豆包 / DeepSeek / Ollama /    │
    │        vLLM                         │
    └─────────────────────────────────────┘
```

---

## 2. 环境准备

### 2.1 系统前置要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.12+ | 推荐使用 pyenv 管理版本 |
| Node.js | 20+ | 前端构建工具 |
| Docker | 24+ | 运行 Qdrant 向量数据库 |
| Docker Compose | 2.20+ | 容器编排 |
| 磁盘空间 | 5 GB+ | Qdrant 数据 + 模型缓存 + 上传文件 |

### 2.2 安装 Python 环境

#### 方式一：使用 pyenv（推荐）

```bash
# macOS
brew install pyenv
pyenv install 3.12.4
pyenv local 3.12.4

# 验证
python --version  # Python 3.12.4
```

#### 方式二：使用 conda

```bash
conda create -n bookrag python=3.12
conda activate bookrag
```

### 2.3 安装 Docker

```bash
# macOS（通过 Homebrew）
brew install --cask docker

# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# 验证
docker --version         # Docker version 24.x.x
docker compose version   # Docker Compose version v2.x.x
```

### 2.4 克隆项目并安装依赖

```bash
# 1. 克隆仓库
git clone <repository-url>
cd book_knowledgebase

# 2. 安装后端依赖
cd backend
pip install -e ".[dev]"
# 若无 pyproject.toml，使用：pip install -r requirements.txt
cd ..

# 3. 安装前端依赖
cd frontend
npm install
cd ..
```

#### FastEmbed 安装注意事项

FastEmbed 用于本地 BM25 稀疏向量计算（进程内 ONNX 推理），安装时会下载 ONNX Runtime：

```bash
pip install fastembed

# 如果网络受限，可指定镜像
pip install fastembed -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装（会触发 BM25 模型下载，约 30MB）
python -c "from fastembed import SparseTextEmbedding; print('FastEmbed OK')"
```

> **注意**：首次运行时 FastEmbed 会自动下载 `Qdrant/bm25` 模型到 `~/.cache/fastembed/`，
> 约 30MB，需要网络连通。离线环境请预先手动下载。

### 2.5 启动 Qdrant

```bash
# 开发模式（后台运行，数据持久化到 ./storage/qdrant）
make dev-qdrant

# 或直接使用 docker compose
docker compose -f docker-compose.yml -f docker-compose.dev.yml up qdrant -d

# 验证 Qdrant 是否正常运行
curl http://localhost:6333/health
# 期望输出：{"title":"qdrant - vector search engine","version":"..."}

# 访问 Qdrant Web UI（可选）
open http://localhost:6333/dashboard
```

### 2.6 初始化数据库

```bash
# 执行 Alembic 数据库迁移
make migrate

# 等价命令
cd backend && alembic upgrade head

# 迁移成功后，SQLite 数据库文件位于：
# backend/storage/app.db（开发环境）
```

---

## 3. 配置说明

### 3.1 创建配置文件

```bash
# 从示例文件复制
cp .env.example .env

# 编辑配置
vim .env  # 或 nano .env / code .env
```

### 3.2 环境变量详细说明

#### 数据库配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/app.db` | 数据库连接串。开发使用 SQLite，生产可改为 PostgreSQL：`postgresql+asyncpg://user:pass@host/db` |

#### Qdrant 向量数据库

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QDRANT_HOST` | `localhost` | Qdrant 服务地址 |
| `QDRANT_PORT` | `6333` | Qdrant gRPC/HTTP 端口 |
| `QDRANT_API_KEY` | （空） | Qdrant Cloud 鉴权 Key，本地部署可留空 |

#### 豆包（火山引擎 Ark）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARK_API_KEY` | **必填** | 火山引擎 Ark API Key，前往 [console.volcengine.com](https://console.volcengine.com) 获取 |
| `ARK_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | Ark API 端点，通常不需修改 |
| `DOUBAO_CHAT_MODEL` | `doubao-seed-2-0-lite-260428` | 豆包对话模型 ID |
| `DOUBAO_TEXT_EMBEDDING_MODEL` | `doubao-embedding-vision-251215` | 豆包文本 Embedding 模型（2048 维） |
| `DOUBAO_VISION_EMBEDDING_MODEL` | `doubao-embedding-vision-251215` | 豆包多模态 Embedding 模型（与文本模型相同，走 `/api/v3/embeddings/multimodal` 端点） |
| `DOUBAO_VISION_CHAT_MODEL` | `doubao-seed-2-0-lite-260428` | 图片 Caption 使用的 Vision LLM |

> **关于豆包 Embedding 端点**：`doubao-embedding-vision-251215` 是多模态模型，
> 文本 Embedding 需要走 `/api/v3/embeddings/multimodal` 端点（而非标准的 `/embeddings`），
> 请求格式为 `{"model": "...", "input": [{"type": "text", "text": "..."}]}`。
> 系统已内置处理此差异，用户无需手动调整，但如遇 API 报错请参考 [§11 常见问题](#11-常见问题排查)。

#### DeepSeek（可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | （空） | DeepSeek API Key，不使用 DeepSeek 可留空 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 端点 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 默认模型，可选 `deepseek-reasoner` |

#### Ollama（本地模型，可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_DEFAULT_CHAT_MODEL` | `llama3` | 默认 Chat 模型 |
| `OLLAMA_DEFAULT_EMBED_MODEL` | `nomic-embed-text` | 默认 Embedding 模型 |

#### vLLM（自托管 OpenAI 兼容接口，可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_BASE_URL` | （空） | vLLM OpenAI 兼容服务地址，如 `http://gpu-server:8001/v1` |
| `VLLM_API_KEY` | `EMPTY` | vLLM 通常不需要真实 Key，保留 `EMPTY` 即可 |

#### RAG 参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | `512` | 文本分块大小（字符数）。技术书推荐 512-800；长文叙事类推荐 800-1200 |
| `CHUNK_OVERLAP` | `64` | 相邻分块重叠字符数，保证上下文连贯性，推荐为 `CHUNK_SIZE` 的 10-15% |
| `DEFAULT_TOP_K` | `8` | 检索返回的 chunk 数量（经 RRF 融合后） |
| `CONTEXT_HISTORY_TURNS` | `10` | 会话历史保留轮数（每轮包含一问一答） |
| `MAX_IMAGE_DIMENSION` | `1024` | 图片预处理最大边长（像素），超出则等比缩放 |

#### 跨域配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | 允许跨域的前端地址，多个用逗号分隔 |

### 3.3 Embedding 模型锁定机制（重要）

这是系统最重要的设计约束，**必须在创建知识库前理解**。

**原理**：每个向量数据库集合（Qdrant Collection）中存储的向量，都是由特定 Embedding 模型生成的。不同 Embedding 模型的向量空间完全不同，若摄入时用模型 A，检索时用模型 B，则计算出的余弦相似度没有意义，会导致**静默的检索失败**（系统不报错，但返回的结果毫不相关）。

**系统行为**：
- 创建知识库时，将 `(provider, model_id, vector_dim)` 三元组永久写入数据库
- 后续每次文档上传和检索查询，系统都会校验三元组是否匹配
- 不匹配时返回 `HTTP 409 Conflict`，明确拒绝操作

```
创建知识库
    │
    ▼
锁定：provider=doubao / model_id=doubao-embedding-vision-251215 / dim=2048
    │
    ├─→ 上传文档 ─→ 校验三元组 ─→ 匹配 → 继续摄入
    │                           └─→ 不匹配 → 409 Conflict
    │
    └─→ 用户提问 ─→ 校验三元组 ─→ 匹配 → 继续检索
                               └─→ 不匹配 → 409 Conflict
```

**实践要点**：
1. 一旦创建知识库，**无法更换 Embedding 模型**
2. 若要切换模型，需删除旧知识库，新建知识库，重新上传所有文档
3. 不同知识库可以使用不同的 Embedding 模型（相互独立）

---

## 4. 快速开始（5 分钟上手）

以下步骤假设你已完成 [§2 环境准备](#2-环境准备) 并配置好 `.env`（至少填写了 `ARK_API_KEY`）。

### 步骤 1：启动所有服务

打开三个终端窗口，分别执行：

```bash
# 终端 1：启动 Qdrant
make dev-qdrant

# 终端 2：启动后端（首次启动会自动执行 DB 迁移）
make dev-backend
# 看到 "Application startup complete." 即为成功

# 终端 3：启动前端
make dev-frontend
# 看到 "Local: http://localhost:5173/" 即为成功
```

### 步骤 2：打开前端，创建知识库

1. 浏览器访问 `http://localhost:5173`
2. 点击左侧侧边栏顶部的 **「+ 新建知识库」** 按钮
3. 填写名称（如：`深度学习教材`），选择 Embedding 提供商 `doubao`
4. 点击 **「创建」**

### 步骤 3：上传电子书

1. 在左侧侧边栏点击刚创建的知识库，展开它
2. 点击 **「上传文档」** 区域，选择一个 PDF 或 EPUB 文件
3. 观察文档状态点变化：灰色（待处理）→ 黄色闪烁（处理中）→ 绿色（已索引）
4. 等待索引完成（取决于文件大小，通常 30 秒 ~ 5 分钟）

### 步骤 4：开始问答

1. 点击知识库旁的 **「新对话」** 按钮
2. 选择 LLM 提供商（如 `doubao`）
3. 在底部输入框输入问题，按回车或点击发送
4. 观察流式回答实时输出

---

## 5. 前端界面使用指南

### 5.1 界面布局

```
┌──────────────────┬────────────────────────────────────┐
│   左侧侧边栏      │          主内容区                   │
│                  │                                    │
│  [+ 新建知识库]  │  ┌──────────────────────────────┐  │
│                  │  │        对话消息区              │  │
│  📚 知识库 A     │  │                              │  │
│    └ 📄 书名.pdf │  │  用户: 什么是反向传播？        │  │
│    └ 📄 书名.epub│  │                              │  │
│                  │  │  助手: 反向传播是...           │  │
│  📚 知识库 B     │  │  [来源: 深度学习.pdf p.42]    │  │
│                  │  │                              │  │
│  ─── 会话列表 ── │  └──────────────────────────────┘  │
│  💬 会话 1       │                                    │
│  💬 会话 2       │  ┌──────────────────────────────┐  │
│                  │  │  [输入问题...]       [发送]   │  │
│                  │  └──────────────────────────────┘  │
└──────────────────┴────────────────────────────────────┘
```

### 5.2 创建知识库

<!-- [截图位置：KBCreateModal 对话框] -->

1. 点击侧边栏顶部的 **「+ 新建知识库」** 按钮
2. 在弹窗中填写以下信息：
   - **知识库名称**（必填）：如 `机器学习教材集`
   - **描述**（可选）：备注用途
   - **Embedding 提供商**：选择 `doubao` / `ollama` / `vllm`
   - **Embedding 模型**：选择后自动填充推荐维度
   - **向量维度**：豆包为 `2048`，nomic-embed-text 为 `768`
3. 点击 **「创建」** 按钮

> **注意**：Embedding 模型一旦选定，**不可更改**。请在创建前确认好要使用的模型。

### 5.3 上传文档

<!-- [截图位置：DocumentUploader 组件] -->

1. 在侧边栏点击目标知识库，展开文档列表
2. 点击文档列表区域的上传框（或拖拽文件到此处）
3. 支持同时选择多个文件（PDF / EPUB / Markdown）
4. 文件上传后立即开始后台处理，状态指示：

| 状态点颜色 | 含义 |
|-----------|------|
| 灰色（静止） | 排队等待处理 |
| 黄色（脉冲闪烁） | 正在解析/向量化 |
| 绿色（静止） | 已成功索引，可检索 |
| 红色（静止） | 处理失败，悬停查看错误信息 |

### 5.4 管理知识库

知识库名称旁的操作按钮：

| 图标 | 功能 | 说明 |
|------|------|------|
| 📊 统计 | 查看 Qdrant 向量点数、文档数 | |
| 🔄 清空 | 删除所有向量，保留元数据 | 需重新上传文档 |
| 🗑️ 删除 | 永久删除知识库及所有数据 | 不可恢复 |

### 5.5 创建对话会话

<!-- [截图位置：ChatWindow + LLM 选择器] -->

1. 在侧边栏的知识库旁，点击 **「+ 新对话」**（或气泡图标）
2. 弹出配置面板，选择：
   - **LLM 提供商**：doubao / deepseek / ollama / vllm
   - **LLM 模型**：下拉选择具体模型
3. 确认后进入对话窗口

### 5.6 对话交互

<!-- [截图位置：MessageBubble 组件] -->

- 底部输入框输入问题，`Enter` 发送（`Shift+Enter` 换行）
- 回答以 **流式方式** 实时渲染（类似 ChatGPT 打字效果）
- 回答下方显示 **参考来源**：文档名 + 页码，可点击查看原文片段
- 对话窗口顶部显示会话绑定的知识库和 LLM 信息

### 5.7 查看可用模型

页面右上角或设置入口可查看当前系统配置的可用 Embedding 模型和 LLM 提供商列表（对应后端接口 `GET /api/v1/embedding-models` 和 `GET /api/v1/llm-providers`）。

---

## 6. CLI 使用指南

CLI 工具基于 Typer + Rich，提供彩色输出和交互式确认。

### 6.1 安装与配置

```bash
# 安装后端包（含 CLI）
cd backend
pip install -e .

# 或直接用 python 调用
python /path/to/book_knowledgebase/cli/main.py --help

# 配置 API 地址（默认 http://localhost:8000/api/v1）
export BOOKRAG_API_URL=http://localhost:8000/api/v1
```

### 6.2 知识库管理命令

#### 列出所有知识库

```bash
python cli/main.py kb list

# 输出示例：
# ┌──────────────────────────────────────┬──────────────┬────────────────────────────────────┬──────┬─────────────────────┐
# │ ID                                   │ 名称         │ Embedding 模型                     │ 维度 │ 创建时间             │
# ├──────────────────────────────────────┼──────────────┼────────────────────────────────────┼──────┼─────────────────────┤
# │ a1b2c3d4-...                         │ 机器学习教材 │ doubao / doubao-embedding-vision.. │ 2048 │ 2026-07-20 10:30:00 │
# └──────────────────────────────────────┴──────────────┴────────────────────────────────────┴──────┴─────────────────────┘
```

#### 创建知识库

```bash
python cli/main.py kb create \
  --name "深度学习书库" \
  --embedding-provider doubao \
  --model doubao-embedding-vision-251215 \
  --dim 2048

# 多模态 Embedding（支持图文联合向量化）
python cli/main.py kb create \
  --name "图文混排书库" \
  --embedding-provider doubao \
  --model doubao-embedding-vision-251215 \
  --dim 2048 \
  --multimodal

# 使用 Ollama 本地模型
python cli/main.py kb create \
  --name "本地知识库" \
  --embedding-provider ollama \
  --model nomic-embed-text \
  --dim 768
```

#### 清空知识库（保留元数据）

```bash
python cli/main.py kb clear --kb-id a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# 跳过交互确认（适合脚本自动化）
python cli/main.py kb clear --kb-id <id> --yes
```

#### 删除知识库（不可恢复）

```bash
python cli/main.py kb delete --kb-id <id>

# 跳过确认
python cli/main.py kb delete --kb-id <id> --yes
```

### 6.3 文档摄入命令

```bash
# 摄入单个 PDF
python cli/main.py ingest path/to/book.pdf --kb-id <kb-id>

# 摄入 EPUB
python cli/main.py ingest path/to/novel.epub --kb-id <kb-id>

# 摄入 Markdown
python cli/main.py ingest path/to/notes.md --kb-id <kb-id>

# 提交后不等待（后台处理，立即返回 doc_id）
python cli/main.py ingest book.pdf --kb-id <kb-id> --no-wait
```

摄入成功后输出示例：

```
╭─────────────────── 摄入成功 ───────────────────╮
│ ✓ 摄入完成                                      │
│ 文档 ID: f7e8d9c0-xxxx-xxxx-xxxx-xxxxxxxxxxxx  │
│ Chunk 数: 342                                   │
│ 图片数: 18                                      │
╰────────────────────────────────────────────────╯
```

### 6.4 检索命令

```bash
# 基础混合检索（不生成回答，仅返回相关 chunk）
python cli/main.py search "什么是注意力机制" --kb-id <kb-id>

# 调整返回数量
python cli/main.py search "Transformer 架构" --kb-id <kb-id> --top-k 10
```

输出示例：

```
查询：什么是注意力机制

╭─ [1] transformer.pdf (页:42) score=0.921 ───────╮
│ 注意力机制（Attention Mechanism）是 Transformer  │
│ 模型的核心组件，通过计算查询向量与键向量的点积...  │
╰──────────────────────────────────────────────────╯
```

### 6.5 对话命令

```bash
# 使用豆包 LLM 进行交互式对话
python cli/main.py chat --kb-id <kb-id> --llm-provider doubao

# 使用 DeepSeek
python cli/main.py chat --kb-id <kb-id> --llm-provider deepseek

# 指定模型版本
python cli/main.py chat --kb-id <kb-id> --llm-provider doubao --llm-model doubao-seed-2-0-lite-260428

# 使用本地 Ollama
python cli/main.py chat --kb-id <kb-id> --llm-provider ollama --llm-model llama3
```

对话模式说明：

```
╭────────────────── 对话 ───────────────────────╮
│ BookRAG 对话模式                               │
│ 知识库: a1b2c3d4... | LLM: doubao             │
│ 输入 exit 或 Ctrl+C 退出                       │
╰────────────────────────────────────────────────╯
会话 ID: 9e8f7a6b-xxxx

[你] 什么是批归一化？

助手：批归一化（Batch Normalization）是一种...
[来源: deep_learning.pdf, neural_networks.pdf]

[你] 它与层归一化有什么区别？

助手：批归一化和层归一化的主要区别在于...

[你] exit
退出对话
```

---

## 7. 知识库管理

### 7.1 知识库生命周期

```
创建知识库（锁定 Embedding 模型）
    │
    ▼
上传文档（可多次追加）
    │
    ▼
建立对话会话（可多个并行）
    │
    ├─→ 定期追加新文档（知识库持续扩充）
    │
    ├─→ 清空向量（保留 KB 元数据，需重新上传）
    │
    └─→ 删除知识库（彻底清除所有数据）
```

### 7.2 创建最佳实践

**命名规范**：建议包含领域和时间信息，如 `机器学习-2026`、`Python编程-进阶`

**选择 Embedding 模型**：

| 场景 | 推荐模型 | 维度 |
|------|----------|------|
| 中文技术书籍（含图） | `doubao-embedding-vision-251215` | 2048 |
| 纯文本中文内容 | `doubao-embedding-vision-251215` | 2048 |
| 本地离线（无 API） | `ollama: nomic-embed-text` | 768 |
| 英文技术文档 | `ollama: nomic-embed-text` 或 doubao | 768 / 2048 |

### 7.3 文档上传最佳实践

**文件格式建议**：
- **PDF**：确保是可选取文字的 PDF（非扫描件）。扫描件需先 OCR 处理
- **EPUB**：标准 EPUB 2/3 格式均支持
- **Markdown**：使用标准 Markdown 语法，图片路径确保可访问

**文件大小**：单文件建议不超过 100MB。超大 PDF（500+ 页）可考虑按章节拆分

**批量上传**：可通过前端同时选择多个文件，系统会依次处理（后台异步任务队列）

### 7.4 监控摄入状态

通过 API 查询知识库统计：

```bash
# 获取知识库统计（向量点数、文档数等）
curl http://localhost:8000/api/v1/knowledge-bases/<kb-id>/stats

# 输出示例：
{
  "kb_id": "a1b2c3d4-...",
  "name": "机器学习教材",
  "document_count": 5,
  "indexed_document_count": 4,
  "failed_document_count": 1,
  "qdrant_points_count": 2847
}
```

### 7.5 清空 vs 删除

| 操作 | 向量数据 | 文档记录 | KB 元数据 | 会话历史 |
|------|----------|----------|-----------|----------|
| 清空（Clear） | 删除 ✗ | 重置为 pending | 保留 ✓ | 保留 ✓ |
| 删除（Delete） | 删除 ✗ | 删除 ✗ | 删除 ✗ | 删除 ✗ |

**使用场景**：
- **清空**：想重新处理所有文档（如升级 chunk_size 参数后），或排查检索质量问题
- **删除**：不再需要该知识库，彻底回收空间

---

## 8. 对话功能使用

### 8.1 会话管理

每个对话会话绑定：
- 一个**知识库**（决定检索范围）
- 一个 **LLM 提供商 + 模型**（决定回答质量和风格）
- **历史记录**（最近 N 轮，由 `CONTEXT_HISTORY_TURNS` 控制，默认 10 轮）

```bash
# API：创建会话
POST /api/v1/sessions
{
  "kb_id": "a1b2c3d4-...",
  "llm_provider": "doubao",
  "llm_model_id": "doubao-seed-2-0-lite-260428"
}

# API：列出某知识库的会话
GET /api/v1/sessions?kb_id=a1b2c3d4-...

# API：获取会话历史消息
GET /api/v1/sessions/{session_id}/messages

# API：删除会话
DELETE /api/v1/sessions/{session_id}
```

### 8.2 流式问答（SSE）

问答接口使用 Server-Sent Events（SSE）实现流式输出：

```bash
# 使用 curl 发起流式问答
curl -N -X POST \
  http://localhost:8000/api/v1/sessions/{session_id}/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "什么是 Transformer 的自注意力机制？"}'

# SSE 事件流格式：
data: {"token": "自"}
data: {"token": "注意"}
data: {"token": "力机制"}
data: {"token": "（Self-Attention）"}
data: {"token": "是"}
...
data: {"done": true, "sources": [{"source_doc": "transformer.pdf", "page_num": 42, "text": "..."}]}
```

**事件字段说明**：

| 字段 | 含义 |
|------|------|
| `token` | LLM 输出的 token 片段（流式） |
| `done` | `true` 表示生成完毕 |
| `sources` | 检索到的来源 chunks，包含文档名、页码、文本摘要 |

### 8.3 多轮对话示例

系统会自动维护对话上下文，支持追问：

```
用户：什么是反向传播算法？
助手：反向传播（Backpropagation）是训练神经网络的核心算法...

用户：它和梯度下降有什么关系？  ← 追问，系统知道"它"指反向传播
助手：反向传播负责计算梯度，梯度下降负责利用这些梯度更新参数...

用户：能举一个具体的数学推导例子吗？  ← 继续追问
助手：好的，以一个两层神经网络为例...
```

### 8.4 理解回答质量

影响回答质量的核心因素：

1. **检索质量**：chunk_size 是否合适，BM25 + Dense 双路是否召回了关键段落
2. **Reranker**：CrossEncoder 重排是否开启（`ENABLE_RERANKER=true`）
3. **LLM 能力**：豆包 seed-2.0 > deepseek-chat > 本地 Ollama（通常）
4. **Prompt 设计**：系统内置 RAG Prompt 模板，仅使用检索到的内容回答，减少幻觉

---

## 9. 数据预处理原理

本节介绍文档从上传到可被检索的完整处理流程。

### 9.1 处理流水线全貌

```
用户上传文件（PDF / EPUB / Markdown）
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 文件解析（Parsers）                          │
│                                                     │
│  PDF ──→ PyMuPDF 逐页提取                           │
│           ├─ 文字块（按段落/行）                     │
│           └─ 图片（过滤 <100×100px，Pillow resize） │
│                                                     │
│  EPUB ─→ ebooklib + BeautifulSoup4                  │
│           ├─ HTML 章节 → 纯文字                     │
│           └─ OPF 资源图片                           │
│                                                     │
│  Markdown → 按 ##/### 标题分段                      │
│           └─ 正则提取 ![alt](path) 图片引用          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: 图片 Caption（ImageCaptioner）               │
│                                                     │
│  图片 ──→ Vision LLM（doubao / llava）               │
│           └─→ 中文描述文字（Caption）                │
│                                                     │
│  Caption chunk 结构：                               │
│  {                                                  │
│    "text": "图片显示一个神经网络结构图...",           │
│    "payload": {                                     │
│      "image_path": "/storage/images/img_001.jpg",  │
│      "chunk_type": "image_caption"                 │
│    }                                               │
│  }                                                  │
│                                                     │
│  优先级：Doubao Vision → Ollama llava → 占位文字    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: 文本分块（RecursiveTextSplitter）            │
│                                                     │
│  按优先级尝试以下分隔符切分：                         │
│  1. \n\n  （段落边界，最佳切分点）                   │
│  2. \n    （换行）                                  │
│  3. 。！？（中文句末标点）                           │
│  4. 空格  （词边界，兜底）                           │
│  5. 字符  （强制切分，最终兜底）                     │
│                                                     │
│  参数：chunk_size=512, chunk_overlap=64             │
│  效果：每块约 512 字符，前后块共享 64 字符重叠       │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: 向量化（Embedding）                          │
│                                                     │
│  每个 chunk 生成两路向量：                           │
│                                                     │
│  Dense Vector（稠密）：                             │
│  └─ Embedding 模型（豆包 / Ollama / vLLM）          │
│     → float32 向量，2048 维（豆包）                  │
│     → 存入 Qdrant "dense" 命名向量                  │
│                                                     │
│  Sparse Vector（稀疏）：                            │
│  └─ FastEmbed BM25（Qdrant/bm25，本地 ONNX）        │
│     → 稀疏权重向量（关键词 TF-IDF）                  │
│     → 存入 Qdrant "sparse" 命名向量                 │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: 写入 Qdrant                                 │
│                                                     │
│  upsert(                                            │
│    id = chunk_uuid,                                 │
│    vectors = {                                      │
│      "dense": [...float32...],                      │
│      "sparse": {indices: [...], values: [...]}      │
│    },                                               │
│    payload = {                                      │
│      "text": "原文内容",                             │
│      "source_doc": "book.pdf",                      │
│      "page_num": 42,                                │
│      "chunk_index": 7,                              │
│      "image_path": null  # 图片 Caption 时有值      │
│    }                                               │
│  )                                                  │
└─────────────────────────────────────────────────────┘
```

### 9.2 检索流程

```
用户提问："什么是 Transformer 的自注意力？"
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 查询向量化                                           │
│  Dense: Embedding 模型 → 2048 维向量                │
│  Sparse: FastEmbed BM25 → 稀疏权重向量              │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Qdrant 混合检索（Prefetch + RRF Fusion）             │
│                                                     │
│  Prefetch Dense:  余弦相似度 → Top-50 候选           │
│  Prefetch Sparse: BM25 分数   → Top-50 候选          │
│         │                  │                        │
│         └──────┬───────────┘                        │
│                ▼                                    │
│          RRF 融合（倒数排名融合）                    │
│          RRF_score = Σ 1/(rank_i + 60)              │
│          → 综合排名 Top-20                          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼ (可选，推荐开启)
┌─────────────────────────────────────────────────────┐
│ CrossEncoder Reranker（精排）                        │
│  FlashRank ms-marco-MiniLM                          │
│  对 (query, chunk) 对重新打分                        │
│  → 最终 Top-K（默认 8）                             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 组装 Prompt + LLM 流式生成                           │
│                                                     │
│  System Prompt：仅基于以下内容回答，如无相关信息则   │
│                 明确告知，不要捏造。                 │
│  Context：Top-K chunks 拼接                         │
│  History：最近 N 轮对话                             │
│  User：当前问题                                     │
└─────────────────────────────────────────────────────┘
```

---

## 10. 模型配置指南

### 10.1 豆包（Doubao）配置

**适用场景**：中文内容理解优秀，API 稳定，推荐作为主要提供商。

```env
# 必填
ARK_API_KEY=your_ark_api_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# Chat 模型
DOUBAO_CHAT_MODEL=doubao-seed-2-0-lite-260428
# 可选其他模型：doubao-pro-4k, doubao-lite-4k 等

# Embedding 模型（2048 维，支持中文）
DOUBAO_TEXT_EMBEDDING_MODEL=doubao-embedding-vision-251215
DOUBAO_VISION_EMBEDDING_MODEL=doubao-embedding-vision-251215

# Vision Caption 模型
DOUBAO_VISION_CHAT_MODEL=doubao-seed-2-0-lite-260428
```

**获取 API Key 步骤**：
1. 访问 [console.volcengine.com](https://console.volcengine.com)
2. 开通「方舟大模型服务」
3. 在「API Key 管理」中创建 Key
4. 确认已开通 `doubao-embedding-vision-251215` 和 `doubao-seed-2-0-lite-260428` 模型权限

### 10.2 DeepSeek 配置

**适用场景**：代码类书籍问答效果好，价格相对低廉。

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
# 推理增强（较慢但更准确）：
# DEEPSEEK_MODEL=deepseek-reasoner
```

**注意**：DeepSeek 没有 Embedding 模型，仅可用作 LLM。Embedding 仍需豆包或 Ollama。

**在前端/CLI 切换到 DeepSeek**：
- 前端：创建会话时选择「LLM 提供商：deepseek」
- CLI：`python cli/main.py chat --kb-id <id> --llm-provider deepseek`

### 10.3 Ollama（本地模型）配置

**适用场景**：完全离线、隐私数据、无 API 费用。

**安装 Ollama**：
```bash
# macOS
brew install ollama

# 或从官网下载
# https://ollama.com/download

# 启动 Ollama 服务
ollama serve
```

**拉取模型**：
```bash
# Chat 模型
ollama pull llama3          # Llama 3 8B（中等质量）
ollama pull qwen2.5:7b      # Qwen 2.5（中文更优）
ollama pull llama3.1:8b     # Llama 3.1 8B

# Embedding 模型
ollama pull nomic-embed-text    # 768 维，通用
ollama pull mxbai-embed-large   # 1024 维，精度更高

# Vision 模型（用于图片 Caption）
ollama pull llava:13b           # LLaVA 13B
```

**配置 .env**：
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_CHAT_MODEL=qwen2.5:7b
OLLAMA_DEFAULT_EMBED_MODEL=nomic-embed-text
```

**创建使用 Ollama Embedding 的知识库**：
```bash
python cli/main.py kb create \
  --name "本地知识库" \
  --embedding-provider ollama \
  --model nomic-embed-text \
  --dim 768
```

### 10.4 vLLM（自托管高性能推理）配置

**适用场景**：企业内网 GPU 服务器，高并发场景，OpenAI 兼容接口。

**启动 vLLM（示例）**：
```bash
# 在 GPU 服务器上（需要 CUDA 环境）
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name qwen2.5-7b
```

**配置 .env**：
```env
VLLM_BASE_URL=http://gpu-server.internal:8001/v1
VLLM_API_KEY=EMPTY
```

**在 CLI 使用 vLLM**：
```bash
python cli/main.py chat \
  --kb-id <id> \
  --llm-provider vllm \
  --llm-model qwen2.5-7b
```

### 10.5 模型提供商对比

| 提供商 | Embedding | Chat/LLM | Vision Caption | 离线 | 成本 |
|--------|-----------|----------|----------------|------|------|
| 豆包（doubao） | ✅ 2048维 | ✅ | ✅ | ✗ | 按量付费 |
| DeepSeek | ✗ | ✅ | ✗ | ✗ | 较低 |
| Ollama | ✅ 768/1024维 | ✅ | ✅ (llava) | ✅ | 免费 |
| vLLM | ✅（依赖部署模型） | ✅ | ✅（依赖模型） | ✅ | GPU 资源 |

---

## 11. 常见问题排查

### 11.1 豆包 API 相关问题

#### 问题：调用豆包 Embedding 报 404 或参数错误

**原因**：`doubao-embedding-vision-251215` 是多模态 Embedding 模型，必须走 `/api/v3/embeddings/multimodal` 端点，请求格式与标准 OpenAI `/embeddings` 不同。

**正确请求格式**：
```python
# 错误（标准 OpenAI 格式，对此模型无效）
{"model": "doubao-embedding-vision-251215", "input": "文本内容"}

# 正确（多模态格式）
{
  "model": "doubao-embedding-vision-251215",
  "input": [
    {"type": "text", "text": "文本内容"}
  ]
}
```

系统已内置处理此差异（在 `backend/core/embedding/doubao_embedding.py` 中），若出现相关报错，请检查 `ARK_BASE_URL` 是否正确设置。

#### 问题：ARK_API_KEY 有效但返回 403

- 检查模型是否已在火山引擎控制台开通权限
- 确认账户余额充足
- 检查 Key 是否有 IP 白名单限制

#### 问题：豆包 Vision Chat（图片 Caption）超时

- 图片 Caption 调用 Vision LLM，单张图片约 3-10 秒
- 对于图片密集的 PDF，总摄入时间会较长（100 张图片约 5-15 分钟）
- 可通过设置较大的超时或使用 `--no-wait` 模式处理

---

### 11.2 FastEmbed 安装问题

#### 问题：`pip install fastembed` 安装失败

```bash
# 常见错误：ONNX Runtime 依赖不满足
# 解决方案 1：指定版本
pip install fastembed==0.3.6

# 解决方案 2：先安装 ONNX Runtime
pip install onnxruntime

# 解决方案 3：使用 CPU 版本
pip install fastembed --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/OnnxRuntime/pypi/simple/
```

#### 问题：首次运行 FastEmbed 时卡住，不下载模型

```bash
# FastEmbed 需要下载 BM25 ONNX 模型（约 30MB）
# 检查网络是否能访问 HuggingFace
curl -I https://huggingface.co

# 如果无法访问，设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 手动触发下载
python -c "
from fastembed import SparseTextEmbedding
model = SparseTextEmbedding('Qdrant/bm25')
print('BM25 模型下载完成')
"
```

---

### 11.3 Qdrant 相关问题

#### 问题：`make dev-qdrant` 后无法连接

```bash
# 检查容器状态
docker ps | grep qdrant

# 查看 Qdrant 日志
docker logs book_knowledgebase-qdrant-1

# 检查端口占用
lsof -i :6333

# 测试连接
curl http://localhost:6333/health
```

#### 问题：Qdrant 版本兼容性

本系统使用 Qdrant Sparse Vector（稀疏向量）特性，**要求 Qdrant >= 1.7.0**。

```bash
# 检查当前版本
curl http://localhost:6333/health | python -m json.tool | grep version

# 如果版本过低，更新 docker-compose.yml 中的 image 版本
# image: qdrant/qdrant:v1.9.0  ← 推荐版本
```

#### 问题：摄入时 Qdrant 返回 "Wrong input: sparse vector"

- 说明 FastEmbed BM25 生成的稀疏向量格式不匹配
- 检查 `fastembed` 版本（推荐 0.3.x）
- 重建 Qdrant Collection：先清空知识库，重新上传文档

---

### 11.4 后端启动问题

#### 问题：`make dev-backend` 报 `ModuleNotFoundError`

```bash
# 确保在正确的虚拟环境中
which python  # 应指向 venv 中的 python

# 重新安装后端依赖
cd backend
pip install -e ".[dev]"
```

#### 问题：数据库迁移失败

```bash
# 查看当前迁移状态
cd backend && alembic current

# 如果迁移版本混乱，可重置（仅开发环境）
rm storage/app.db
alembic upgrade head
```

#### 问题：端口 8000 被占用

```bash
# 查找占用进程
lsof -i :8000

# 终止占用进程
kill -9 <PID>

# 或使用其他端口
uvicorn main:app --port 8080 --reload
# 同时修改 .env: CORS_ORIGINS=http://localhost:5173
```

---

### 11.5 前端相关问题

#### 问题：前端无法连接后端（CORS 错误）

检查 `.env` 中的 `CORS_ORIGINS` 是否包含前端地址：
```env
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

#### 问题：SSE 流式回答不显示

- 确认使用的是 `http://`（非 `https://`）在开发环境
- 检查浏览器是否屏蔽了 EventSource
- 在 Network 面板查看 `/chat/stream` 请求的 Response Headers 是否包含 `Content-Type: text/event-stream`

---

### 11.6 摄入质量问题

#### 问题：摄入失败，文档状态变红

```bash
# 查看详细错误
curl http://localhost:8000/api/v1/knowledge-bases/<kb-id>/documents
# 找到对应文档的 error_message 字段

# 常见原因：
# - PDF 加密：需先解密 PDF
# - 文件编码问题：非 UTF-8 的 Markdown 文件需转码
# - Qdrant 未启动：make dev-qdrant 并等待就绪
# - ARK_API_KEY 未配置或无效：图片 Caption 失败会导致整体失败
```

#### 问题：检索结果质量差，回答不相关

1. **检查 chunk_size**：对于技术书籍，512 可能过小，尝试 768-1024
2. **确认 Reranker 开启**（默认开启）
3. **验证 Embedding 模型一致性**：摄入和检索需使用同一模型（系统会自动校验）
4. **增加 top_k**：在 `.env` 中调整 `DEFAULT_TOP_K=12`

---

## 12. 高级配置

### 12.1 分块参数调优

`chunk_size` 和 `chunk_overlap` 是影响检索质量最关键的参数，需要根据内容类型调整：

| 内容类型 | 推荐 chunk_size | 推荐 chunk_overlap | 原因 |
|----------|-----------------|-------------------|------|
| 技术教材（概念密集） | 512-768 | 64-100 | 概念完整性，避免跨块截断 |
| 文学/叙事类 | 800-1200 | 100-150 | 保留叙事连贯性 |
| 学术论文 | 400-600 | 50-80 | 摘要/结论通常较短 |
| 代码文档 | 300-500 | 30-50 | 代码块通常完整 |
| 问答类（FAQ） | 200-400 | 20-40 | 每条 QA 独立 |

**修改方式**：

```env
# .env
CHUNK_SIZE=768
CHUNK_OVERLAP=100
```

> **注意**：修改 chunk 参数后，需要**清空并重新摄入**已有知识库，否则新旧 chunk 大小不一致会影响检索一致性。

### 12.2 检索参数调优

#### top_k 调整

`DEFAULT_TOP_K` 控制最终传递给 LLM 的上下文 chunk 数量：

```env
DEFAULT_TOP_K=8   # 默认值，平衡质量和速度
DEFAULT_TOP_K=12  # 更多上下文，适合复杂问题，但 LLM 输入 token 增加
DEFAULT_TOP_K=5   # 更精准，适合简单事实查询
```

#### Reranker 配置

重排（rerank）在 RRF 融合之后对候选 chunk 精排，可显著提升相关性。**默认开启**，
前端输入框右下角的「重排」按钮可以按请求关闭。

```env
# .env
RERANK_ENABLED=true                       # 全局默认开关
RERANK_BACKEND=http                       # http = vLLM /v1/rerank；flashrank = 本地 ONNX
RERANK_BASE_URL=http://localhost:8100/v1  # 端口即 vLLM 监听端口
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATES=30                      # 参与精排的候选数（>= top_k）
RERANK_TIMEOUT=30
RERANK_AUTOSTART=false                    # 后端启动时是否顺带拉起 vLLM
```

说明：

- `rerank_candidates` 只有大于 `top_k` 才有意义：系统会按它放大召回池，精排后再截断到 `top_k`。
- rerank 服务不可用时自动回退到 RRF 顺序（日志 `rerank_failed`，60 秒内不再重试），不影响问答。
- 想完全本地化、不启 vLLM 时：`RERANK_BACKEND=flashrank`（模型由 `RERANK_LOCAL_MODEL` 指定，偏英文场景）。

vLLM 由项目统一管理（不需要记 `vllm serve` 的一长串参数）：

```bash
make rerank-up        # 启动并等待健康检查
make rerank-status    # 状态 / PID / 健康探针（也能发现手工启动的实例）
make rerank-logs      # 日志尾部
make rerank-down      # 停止
```

- 直接调模块：`cd backend && python -m core.rerank_server up|down|status|logs`
- HTTP：`GET /api/v1/rerank`、`POST /api/v1/rerank/start|stop|restart`、`GET /api/v1/rerank/logs`
- PID / 日志：`storage/run/rerank-server.pid`、`storage/logs/rerank-server.log`
- WSL 提示：mirrored 网络下 `127.0.0.1` 转发可能不稳定，把 `RERANK_BASE_URL` 的 host 换成本机 IP 即可

检索内部链路的参数（通过代码修改 `backend/core/retrieval/`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prefetch_limit` | `max(20, top_k * 2)` | Qdrant 每路（Dense/Sparse）预取数量 |
| `rrf_rank_constant` | 60 | RRF 公式中的排名常数 k（标准值为 60） |

### 12.3 会话历史配置

```env
CONTEXT_HISTORY_TURNS=10   # 保留最近 10 轮对话（20 条消息）
# 增加轮数会增加每次请求的 token 消耗
# 减少轮数可降低成本，适合无需追问的场景
```

### 12.4 图片处理配置

```env
MAX_IMAGE_DIMENSION=1024   # 图片预处理最大边长
# 减小此值可加快处理速度，但可能影响 Caption 质量
# 增大此值（如 2048）可提高 Caption 准确性，但增加 API 调用成本
```

**禁用图片 Caption**（纯文字知识库）：

在 `backend/core/captioning/` 中，若 `ARK_API_KEY` 未配置且 Ollama 未运行 Vision 模型，系统会自动降级为占位文字（`[图片：无法生成描述]`），不影响文字内容的检索。

### 12.5 生产环境部署建议

**切换到 PostgreSQL**：
```env
DATABASE_URL=postgresql+asyncpg://bookrag:password@postgres-host:5432/bookrag_db
```

**Qdrant Cloud 配置**：
```env
QDRANT_HOST=your-cluster.aws.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_cloud_api_key
```

**使用 Docker Compose 完整部署**：
```bash
# 生产构建
make build

# 启动所有服务（含后端容器）
make up

# 查看日志
make logs-backend
```

**Nginx 反向代理配置**（前端生产构建）：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /var/www/bookrag/dist;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 流式响应必须禁用缓冲
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

### 12.6 批量摄入脚本

```bash
# 使用 Makefile 批量摄入整个目录
make ingest path=./my_books/ kb=<kb-id>

# 或自定义脚本
#!/bin/bash
KB_ID="your-kb-id"
BOOKS_DIR="./books"

for file in "$BOOKS_DIR"/*.{pdf,epub,md}; do
    if [ -f "$file" ]; then
        echo "Ingesting: $file"
        python cli/main.py ingest "$file" --kb-id "$KB_ID"
        sleep 2  # 避免并发过多
    fi
done
```

---

## 附录：API 快速参考

### 知识库接口

```
POST   /api/v1/knowledge-bases              创建知识库
GET    /api/v1/knowledge-bases              列出所有知识库
GET    /api/v1/knowledge-bases/{id}         获取知识库详情
DELETE /api/v1/knowledge-bases/{id}         删除知识库
POST   /api/v1/knowledge-bases/{id}/clear   清空向量数据
GET    /api/v1/knowledge-bases/{id}/stats   获取统计信息
```

### 文档接口

```
POST   /api/v1/knowledge-bases/{id}/documents              上传文档
GET    /api/v1/knowledge-bases/{id}/documents              列出文档
GET    /api/v1/knowledge-bases/{id}/documents/{doc_id}     获取文档
DELETE /api/v1/knowledge-bases/{id}/documents/{doc_id}     删除文档
GET    /api/v1/knowledge-bases/{id}/documents/{doc_id}/status 查询摄入状态
```

### 检索接口

```
POST   /api/v1/knowledge-bases/{id}/search  混合检索（不生成回答）
```

### 会话与对话接口

```
POST   /api/v1/sessions                       创建会话
GET    /api/v1/sessions?kb_id={id}            列出会话
DELETE /api/v1/sessions/{session_id}          删除会话
GET    /api/v1/sessions/{session_id}/messages 获取历史消息
POST   /api/v1/sessions/{session_id}/chat/stream  SSE 流式问答
```

### 配置查询接口

```
GET    /api/v1/embedding-models    列出可用 Embedding 模型
GET    /api/v1/llm-providers       列出可用 LLM 提供商
```

---

> 如有问题，请查阅 [系统架构文档](ARCHITECTURE.md) 或 [数据预处理指南](preprocessing_guide.md)。
> 微调功能详见 [微调指南](finetuning_guide.md)。
