# 电子书 RAG 知识库系统 — 使用手册

> 版本：v0.1.0 | 更新日期：2026-07-22

---

## 目录

1. [快速开始（5 分钟跑起来）](#1-快速开始5-分钟跑起来)
2. [详细配置说明](#2-详细配置说明)
3. [前端使用指南](#3-前端使用指南)
4. [CLI 完整用法](#4-cli-完整用法)
5. [API 参考](#5-api-参考)
6. [数据预处理流程说明](#6-数据预处理流程说明)
7. [多模型配置指南](#7-多模型配置指南)
8. [常见问题排查（FAQ）](#8-常见问题排查faq)
9. [生产部署建议](#9-生产部署建议)

---

## 1. 快速开始（5 分钟跑起来）

### 1.1 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.12+ | 后端运行时 |
| Node.js | 18+ | 前端构建 |
| Docker | 24+ | 运行 Qdrant |
| Docker Compose | v2 | 容器编排 |

### 1.2 克隆并配置环境变量

```bash
# 进入项目目录
cd book_knowledgebase

# 复制环境变量模板
cp .env.example .env
```

打开 `.env`，**至少**填写以下两项：

```bash
# 豆包 API Key（用于 Embedding 和问答）
ARK_API_KEY=your_ark_api_key_here

# 其余配置保持默认即可快速运行
```

> 如果没有豆包 API Key，可先用 Ollama 本地运行，详见 [第 7 节](#7-多模型配置指南)。

### 1.3 安装后端依赖

```bash
# 安装 Python 依赖
make install

# 或手动安装
cd backend && pip install -r requirements.txt
```

### 1.4 安装前端依赖

```bash
make install-frontend
# 或
cd frontend && npm install
```

### 1.5 启动服务（三个终端）

**终端 1：启动 Qdrant 向量数据库**

```bash
make dev-qdrant
# 等待出现 "Qdrant is ready!" 后继续
```

**终端 2：启动后端**

```bash
make dev-backend
# 后端监听 http://localhost:8000
# 访问 http://localhost:8000/docs 查看 API 文档（Swagger UI）
```

**终端 3：启动前端**

```bash
make dev-frontend
# 前端监听 http://localhost:5173
```

### 1.6 运行数据库迁移（首次启动必须执行）

```bash
make migrate
# 输出类似：INFO  [alembic.runtime.migration] Running upgrade -> xxxxxx
```

### 1.7 验证服务正常

```bash
curl http://localhost:8000/api/v1/health
# 期望返回：{"status":"ok","qdrant":"connected"}
```

打开浏览器访问 `http://localhost:5173`，页面正常加载即表示全部服务就绪。

### 1.8 创建第一个知识库并上传书籍

```bash
# 创建知识库（使用豆包 embedding）
python cli/main.py kb create \
  --name "我的书库" \
  --embedding-provider doubao \
  --model doubao-embedding-vision-251215 \
  --dim 2048

# 记录输出中的 ID，例如：3f8a1b2c-...

# 上传一本书
python cli/main.py ingest ./my_book.pdf --kb-id 3f8a1b2c-...

# 开始提问
python cli/main.py chat --kb-id 3f8a1b2c-...
```

---

## 2. 详细配置说明

所有配置通过项目根目录的 `.env` 文件设置。系统使用 `pydantic-settings` 加载，环境变量名不区分大小写。

### 2.1 数据库配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/app.db` | 数据库连接串。开发环境使用 SQLite，生产环境推荐 PostgreSQL |

**切换到 PostgreSQL：**

```bash
DATABASE_URL=postgresql+asyncpg://bookrag:bookrag@localhost:5432/bookrag
```

### 2.2 Qdrant 向量库配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `QDRANT_HOST` | `localhost` | Qdrant 服务地址。Docker Compose 生产模式下应填 `qdrant` |
| `QDRANT_PORT` | `6333` | Qdrant HTTP 端口 |
| `QDRANT_API_KEY` | 空 | Qdrant 访问密钥，部署到公网时必须设置 |

### 2.3 存储路径

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STORAGE_ROOT` | `./storage` | 文件存储根目录。包含 `docs/`（上传的原始文件）和 `images/`（提取的图片）两个子目录 |

> 生产部署时建议挂载独立磁盘到 `STORAGE_ROOT` 路径，避免容器重启丢失数据。

### 2.4 豆包 / 火山引擎 Ark 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ARK_API_KEY` | 空（**必填**） | 火山引擎 Ark 平台 API Key。前往 [火山引擎控制台](https://console.volcengine.com/ark) 获取 |
| `ARK_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | API 基础地址，通常无需修改 |
| `DOUBAO_CHAT_MODEL` | `doubao-seed-2-0-lite-260428` | 默认问答模型 ID |
| `DOUBAO_TEXT_EMBEDDING_MODEL` | `doubao-embedding-vision-251215` | 文本 Embedding 模型 ID |
| `DOUBAO_VISION_EMBEDDING_MODEL` | `doubao-embedding-vision-251215` | 多模态 Embedding 模型 ID（同时支持文字和图片） |
| `DOUBAO_VISION_CHAT_MODEL` | `doubao-seed-2-0-lite-260428` | 图片 Caption 所用的视觉模型 |

> **重要**：`doubao-embedding-vision-251215` 走 `/embeddings/multimodal` 接口，每条输入格式为 `{"type":"text","text":"..."}` 或 `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` ，与标准 OpenAI embeddings 接口不同，系统已内置适配。

### 2.5 DeepSeek 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key，填写后前端会出现 DeepSeek 提供商选项 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 默认使用的 DeepSeek 模型 |

### 2.6 Ollama 本地模型配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_DEFAULT_CHAT_MODEL` | `llama3` | 默认问答模型 |
| `OLLAMA_DEFAULT_EMBED_MODEL` | `nomic-embed-text` | 默认 Embedding 模型（维度 768） |

### 2.7 vLLM 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `VLLM_BASE_URL` | 空 | vLLM 服务地址，例如 `http://gpu-server:8080/v1`。填写后启用 vLLM 提供商 |
| `VLLM_API_KEY` | `EMPTY` | vLLM 通常不需要 API Key，保持默认即可 |

### 2.8 RAG 默认参数

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEFAULT_TOP_K` | `8` | 默认检索返回的 chunk 数量。值越大召回率越高，但响应速度越慢 |
| `CONTEXT_HISTORY_TURNS` | `10` | 多轮对话携带的历史轮数。超出部分自动丢弃 |
| `CHUNK_SIZE` | `512` | 文本分块大小（按字符数）。技术书籍推荐 512-800 |
| `CHUNK_OVERLAP` | `64` | 相邻 chunk 的重叠字符数，避免语义在块边界断裂 |
| `MAX_IMAGE_DIMENSION` | `1024` | 提取图片的最大边长（像素），超出则自动缩放 |

### 2.9 Rerank 重排

重排默认开启，检索时先召回 `RERANK_CANDIDATES` 条候选，再用 rerank 模型精排并截断到 `top_k`。
前端输入框的「重排」按钮可按请求关闭。

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `RERANK_ENABLED` | `true` | 全局默认开关；请求里的 `use_reranker` 优先 |
| `RERANK_BACKEND` | `http` | `http` = vLLM `/v1/rerank`；`flashrank` = 本地 ONNX |
| `RERANK_BASE_URL` | `http://localhost:8100/v1` | rerank 服务地址（端口即 vLLM 监听端口）|
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | rerank 模型名 |
| `RERANK_CANDIDATES` | `30` | 参与精排的候选数（应不小于 `top_k`）|
| `RERANK_TIMEOUT` | `30` | rerank 请求超时（秒）|
| `RERANK_LOCAL_MODEL` | `ms-marco-MiniLM-L-6-v2` | `flashrank` 后端的本地模型 |
| `RERANK_AUTOSTART` | `false` | 后端启动时是否顺带拉起 vLLM（`--reload` 开发模式不建议开）|
| `RERANK_SERVER_BIN` | `vllm` | vLLM 可执行文件，可填绝对路径 |
| `RERANK_SERVER_BIND` | `0.0.0.0` | vLLM 监听地址（WSL 下绑 127.0.0.1 可能连不上）|
| `RERANK_SERVER_EXTRA_ARGS` | `--gpu-memory-utilization 0.3 --max-model-len 2048` | 追加给 `vllm serve` 的参数 |
| `RERANK_START_TIMEOUT` | `600` | 启动后等待健康检查的最长秒数 |

#### 服务生命周期

vLLM 由本项目统一管理（PID 与日志在 `storage/run/`、`storage/logs/`）：

```bash
make rerank-up        # 启动并等待健康
make rerank-status    # 状态 + 健康探针 + PID
make rerank-logs      # 日志尾部
make rerank-down      # 停止
```

也可用 `cd backend && python -m core.rerank_server up|down|status|logs`，
或 HTTP 接口 `/api/v1/rerank`（`start` / `stop` / `restart` / `logs`）。

> WSL 注意：mirrored 网络下 `127.0.0.1` 的端口转发可能不稳定（连接被拒），
> 但本机网卡 IP 可用。此时把 `RERANK_BASE_URL` 改成 `http://<本机IP>:8100/v1`
> （`hostname -I` 查看），换网络后需同步更新。

rerank 服务不可用时自动回退到 RRF 顺序，并在日志里打 `rerank_failed`，不影响问答。

### 2.10 CORS 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | 允许跨域的前端地址，多个用逗号分隔 |

---

## 3. 前端使用指南

前端为单页应用（SPA），访问地址 `http://localhost:5173`（开发）或 `http://your-server`（生产）。

### 3.1 整体布局

界面分为三栏：

```
┌─────────────────┬──────────────────────────────┬──────────────────┐
│  左侧边栏        │         主内容区               │  右侧面板（可选）  │
│  ─────────────  │  ──────────────────────────  │  ────────────── │
│  知识库列表       │  对话历史 / 检索结果             │  来源文档预览     │
│  ─────────────  │                              │  图片展示        │
│  会话列表         │  消息输入框                    │                 │
└─────────────────┴──────────────────────────────┴──────────────────┘
```

### 3.2 创建知识库

1. 点击左侧边栏顶部的「+ 新建知识库」按钮
2. 填写知识库名称（必填）和描述（选填）
3. 选择 Embedding 提供商和模型：
   - **豆包**：选 `doubao-embedding-vision-251215`，维度自动填写 `2048`
   - **Ollama**：选 `nomic-embed-text`，维度填写 `768`
4. 如需多模态支持（图文混排书籍），勾选「多模态 Embedding」
5. 点击「创建」

> 注意：知识库创建后 Embedding 模型无法更改。如果要换模型，需删除并重新创建知识库。

### 3.3 上传电子书

1. 在左侧边栏点击目标知识库，展开后显示文档列表
2. 点击「上传文档」或将文件拖入上传区域
3. 支持同时选择多个文件（PDF / EPUB / Markdown）
4. 上传后文件进入处理队列，状态图标说明：
   - 灰色圆点：等待处理（`pending`）
   - 黄色脉冲动画：处理中（`processing`）
   - 绿色圆点：已完成索引（`indexed`）
   - 红色圆点：处理失败（`failed`），鼠标悬停查看错误信息
5. 处理完成后，右侧会显示该文档的 chunk 数和图片数

> 大型 PDF（200 页以上）处理时间可能需要 1-5 分钟，页面可以关闭后台继续处理。

### 3.4 开始对话

1. 点击左侧边栏的「+ 新建会话」（或点击已有会话继续）
2. 选择当前会话使用的 LLM 提供商和模型：
   - **豆包**：需填写 `ARK_API_KEY`
   - **DeepSeek**：需填写 `DEEPSEEK_API_KEY`
   - **Ollama**：本地服务，无需 API Key
3. 在底部输入框输入问题，按 Enter 或点击发送按钮
4. 助手回答以流式方式逐字输出（SSE）
5. 回答结束后，右侧面板展示引用的文档来源，包含：
   - 来源文件名和页码
   - 相关度分数
   - 原始文字片段
   - 若来源是图片，展示图片缩略图

### 3.5 高级检索设置

在对话输入框旁点击「设置」图标，可调整：

| 参数 | 说明 |
|------|------|
| Top-K | 检索返回的 chunk 数量（1-50，默认 8） |
| Score Threshold | 最低相似度阈值（0.0-1.0，默认 0.0 不过滤） |
| 启用 Reranker | 使用 CrossEncoder 对结果二次排序（质量更高，速度稍慢） |

### 3.6 知识库管理

在左侧边栏，每个知识库右侧有两个操作图标：

- **刷新图标（清空向量）**：删除所有向量数据，保留文档记录，可重新触发摄入
- **删除图标（删除知识库）**：永久删除知识库、所有文档记录、向量数据和会话历史

---

## 4. CLI 完整用法

CLI 工具位于 `cli/main.py`，使用前确保后端服务运行中（默认连接 `http://localhost:8000/api/v1`）。

如需连接其他地址，设置环境变量：

```bash
export BOOKRAG_API_URL=http://your-server:8000/api/v1
```

### 4.1 知识库管理

#### 列出所有知识库

```bash
python cli/main.py kb list
```

示例输出：

```
                          知识库列表
┌──────────────────────────────────────┬──────────────┬─────────────────────────────────────────┬──────┬─────────────────────┐
│ ID                                   │ 名称         │ Embedding 模型                          │ 维度 │ 创建时间            │
├──────────────────────────────────────┼──────────────┼─────────────────────────────────────────┼──────┼─────────────────────┤
│ 3f8a1b2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c │ ML 教材库     │ doubao / doubao-embedding-vision-251215 │ 2048 │ 2026-07-20 14:32:11 │
│ a1b2c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d │ Python 书库   │ ollama / nomic-embed-text              │  768 │ 2026-07-21 09:15:43 │
└──────────────────────────────────────┴──────────────┴─────────────────────────────────────────┴──────┴─────────────────────┘
```

#### 创建知识库

```bash
python cli/main.py kb create \
  --name "深度学习书库" \
  --embedding-provider doubao \
  --model doubao-embedding-vision-251215 \
  --dim 2048 \
  --multimodal \
  --desc "深度学习相关书籍和论文"
```

参数说明：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--name` | 是 | 知识库名称 |
| `--embedding-provider` | 是 | `doubao` / `ollama` / `vllm` |
| `--model` | 是 | Embedding 模型 ID |
| `--dim` | 是 | 向量维度（doubao 为 2048，ollama nomic-embed-text 为 768） |
| `--multimodal` | 否 | 启用多模态 embedding，处理含图片的书籍时建议开启 |
| `--desc` | 否 | 知识库描述 |

示例输出：

```
╭─────────────────── 知识库创建 ───────────────────╮
│ ✓ 知识库创建成功                                  │
│ ID: 7c8d9e0f-1a2b-3c4d-5e6f-7a8b9c0d1e2f        │
│ 名称: 深度学习书库                                 │
│ Embedding: doubao / doubao-embedding-vision-251215 (dim=2048) │
╰──────────────────────────────────────────────────╯
```

#### 清空知识库向量

```bash
# 需要二次确认
python cli/main.py kb clear --kb-id 3f8a1b2c-...

# 跳过确认（脚本自动化时使用）
python cli/main.py kb clear --kb-id 3f8a1b2c-... --yes
```

#### 删除知识库

```bash
# 危险操作，永久删除所有数据
python cli/main.py kb delete --kb-id 3f8a1b2c-...

# 跳过确认
python cli/main.py kb delete --kb-id 3f8a1b2c-... --yes
```

### 4.2 文档摄入

```bash
# 摄入 PDF
python cli/main.py ingest ./deep_learning.pdf --kb-id 3f8a1b2c-...

# 摄入 EPUB
python cli/main.py ingest ./python_cookbook.epub --kb-id 3f8a1b2c-...

# 摄入 Markdown
python cli/main.py ingest ./notes.md --kb-id 3f8a1b2c-...

# 摄入后不等待完成（异步，立即返回）
python cli/main.py ingest ./big_book.pdf --kb-id 3f8a1b2c-... --no-wait
```

摄入成功示例输出：

```
╭──────────────── 摄入成功 ────────────────╮
│ ✓ 摄入完成                               │
│ 文档 ID: d4e5f6a7-...                   │
│ Chunk 数: 342                            │
│ 图片数: 28                               │
╰─────────────────────────────────────────╯
```

### 4.3 检索（不生成回答）

```bash
python cli/main.py search "什么是反向传播算法" --kb-id 3f8a1b2c-...

# 自定义返回数量
python cli/main.py search "梯度消失问题" --kb-id 3f8a1b2c-... --top-k 3
```

示例输出：

```
查询： 什么是反向传播算法

╭─ [1] deep_learning.pdf (页:147) score=0.8923 ──────────────────╮
│ 反向传播算法（Backpropagation）是训练神经网络的核心算法。它通过   │
│ 链式法则计算损失函数对每个参数的梯度，然后使用梯度下降法更新参数。  │
╰──────────────────────────────────────────────────────────────────╯

╭─ [2] neural_networks.pdf (页:89) score=0.8441 ─────────────────╮
│ 反向传播的关键思想是将输出误差逐层向后传递，每一层根据其对误差的   │
│ 贡献程度调整权重...                                              │
╰──────────────────────────────────────────────────────────────────╯
```

### 4.4 交互式对话

```bash
# 使用豆包（默认）
python cli/main.py chat --kb-id 3f8a1b2c-...

# 使用 DeepSeek
python cli/main.py chat --kb-id 3f8a1b2c-... --llm-provider deepseek

# 使用 Ollama 本地模型
python cli/main.py chat --kb-id 3f8a1b2c-... --llm-provider ollama --llm-model qwen2.5:7b
```

进入对话模式：

```
╭────────────────────── 对话 ──────────────────────────╮
│ BookRAG 对话模式                                      │
│ 知识库: 3f8a1b2c-... | LLM: doubao                   │
│ 输入 exit 或 Ctrl+C 退出                              │
╰──────────────────────────────────────────────────────╯
会话 ID: 9a0b1c2d-...

[你] 请解释一下注意力机制

助手：注意力机制（Attention Mechanism）是深度学习中的一种技术，
它允许模型在处理序列数据时，有选择性地关注输入序列的不同部分...

来源: attention_paper.pdf, transformer_book.pdf

[你] exit
退出对话
```

---

## 5. API 参考

所有接口基础路径：`http://localhost:8000/api/v1`

完整交互式文档（Swagger UI）：`http://localhost:8000/docs`

### 5.1 知识库接口

#### 创建知识库

```http
POST /api/v1/knowledge-bases
Content-Type: application/json

{
  "name": "深度学习书库",
  "description": "深度学习相关书籍",
  "embedding_provider": "doubao",
  "embedding_model_id": "doubao-embedding-vision-251215",
  "embedding_vector_dim": 2048,
  "embedding_is_multimodal": true,
  "caption_provider": "doubao",
  "caption_model_id": "doubao-seed-2-0-lite-260428"
}
```

响应（201 Created）：

```json
{
  "id": "3f8a1b2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "name": "深度学习书库",
  "description": "深度学习相关书籍",
  "embedding_provider": "doubao",
  "embedding_model_id": "doubao-embedding-vision-251215",
  "embedding_vector_dim": 2048,
  "embedding_is_multimodal": true,
  "caption_provider": "doubao",
  "caption_model_id": "doubao-seed-2-0-lite-260428",
  "qdrant_collection_name": "kb_3f8a1b2c4d5e6f7a8b9c0d1e2f3a4b5c",
  "created_at": "2026-07-22T10:00:00.000Z",
  "updated_at": "2026-07-22T10:00:00.000Z"
}
```

#### 列出所有知识库

```http
GET /api/v1/knowledge-bases
```

响应（200 OK）：

```json
[
  {
    "id": "3f8a1b2c-...",
    "name": "深度学习书库",
    ...
  }
]
```

#### 获取知识库统计

```http
GET /api/v1/knowledge-bases/{kb_id}/stats
```

响应（200 OK）：

```json
{
  "total_docs": 5,
  "indexed_docs": 4,
  "failed_docs": 1,
  "total_chunks": 1842,
  "total_images": 136,
  "qdrant_points_count": 1978
}
```

#### 清空知识库向量

```http
POST /api/v1/knowledge-bases/{kb_id}/clear
```

响应（200 OK）：

```json
{
  "message": "Knowledge base '深度学习书库' cleared successfully",
  "kb_id": "3f8a1b2c-..."
}
```

#### 删除知识库

```http
DELETE /api/v1/knowledge-bases/{kb_id}
```

响应：204 No Content

### 5.2 文档接口

#### 上传文档

```http
POST /api/v1/knowledge-bases/{kb_id}/documents
Content-Type: multipart/form-data

file=@deep_learning.pdf
```

响应（201 Created）：

```json
{
  "id": "d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a",
  "kb_id": "3f8a1b2c-...",
  "filename": "deep_learning.pdf",
  "file_type": "pdf",
  "file_size_bytes": 15728640,
  "status": "pending",
  "error_message": null,
  "chunk_count": 0,
  "image_count": 0,
  "doc_metadata": null,
  "processing_started_at": null,
  "processing_completed_at": null,
  "created_at": "2026-07-22T10:05:00.000Z"
}
```

#### 查询文档状态

```http
GET /api/v1/knowledge-bases/{kb_id}/documents/{doc_id}/status
```

响应（200 OK）：

```json
{
  "id": "d4e5f6a7-...",
  "status": "indexed",
  "chunk_count": 342,
  "image_count": 28,
  "error_message": null
}
```

`status` 取值：`pending` | `processing` | `indexed` | `failed`

#### 列出知识库的所有文档

```http
GET /api/v1/knowledge-bases/{kb_id}/documents
```

#### 删除文档

```http
DELETE /api/v1/knowledge-bases/{kb_id}/documents/{doc_id}
```

响应：204 No Content

### 5.3 会话与对话接口

#### 创建会话

```http
POST /api/v1/sessions
Content-Type: application/json

{
  "kb_id": "3f8a1b2c-...",
  "llm_provider": "doubao",
  "llm_model_id": "doubao-seed-2-0-lite-260428",
  "llm_config": {
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "title": "深度学习问答"
}
```

响应（201 Created）：

```json
{
  "id": "9a0b1c2d-3e4f-5a6b-7c8d-9e0f1a2b3c4d",
  "kb_id": "3f8a1b2c-...",
  "title": "深度学习问答",
  "llm_provider": "doubao",
  "llm_model_id": "doubao-seed-2-0-lite-260428",
  "llm_config": {"temperature": 0.7, "max_tokens": 2048},
  "created_at": "2026-07-22T10:10:00.000Z",
  "updated_at": "2026-07-22T10:10:00.000Z"
}
```

#### 流式问答（SSE）

```http
POST /api/v1/sessions/{session_id}/chat/stream
Content-Type: application/json

{
  "content": "请解释注意力机制的工作原理",
  "image": null,
  "retrieval_config": {
    "top_k": 8,
    "score_threshold": 0.0,
    "use_reranker": false
  }
}
```

响应为 `text/event-stream`，逐行推送 JSON 事件：

```
data: {"token": "注意力"}

data: {"token": "机制"}

data: {"token": "（Attention"}

data: {"token": " Mechanism）"}

...

data: {"done": true, "message_id": "msg-uuid", "sources": [
  {
    "chunk_id": "chunk-uuid",
    "text": "注意力机制允许模型关注...",
    "score": 0.8923,
    "source_doc": "attention_is_all_you_need.pdf",
    "doc_id": "doc-uuid",
    "page_num": 3,
    "chunk_type": "text",
    "image_path": null
  }
]}
```

**携带图片查询（多模态）：**

```json
{
  "content": "这张图表说明的是什么？",
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "retrieval_config": {"top_k": 5}
}
```

#### 获取会话历史消息

```http
GET /api/v1/sessions/{session_id}/messages
```

响应（200 OK）：

```json
[
  {
    "id": "msg-uuid-1",
    "session_id": "9a0b1c2d-...",
    "role": "user",
    "content": "请解释注意力机制的工作原理",
    "image_paths": null,
    "sources": null,
    "prompt_tokens": null,
    "completion_tokens": null,
    "created_at": "2026-07-22T10:11:00.000Z"
  },
  {
    "id": "msg-uuid-2",
    "role": "assistant",
    "content": "注意力机制（Attention Mechanism）...",
    "sources": [...],
    ...
  }
]
```

### 5.4 模型信息接口

#### 查询可用 Embedding 模型

```http
GET /api/v1/embedding-models
```

响应（200 OK）：

```json
[
  {
    "provider": "doubao",
    "model_id": "doubao-embedding-vision-251215",
    "vector_dim": 2048,
    "supports_multimodal": true,
    "description": "Doubao Embedding (doubao-embedding-vision-251215)"
  },
  {
    "provider": "ollama",
    "model_id": "nomic-embed-text",
    "vector_dim": 768,
    "supports_multimodal": false,
    "description": "Ollama nomic-embed-text"
  }
]
```

#### 查询可用 LLM 提供商

```http
GET /api/v1/llm-providers
```

#### 健康检查

```http
GET /api/v1/health
```

响应（200 OK）：

```json
{
  "status": "ok",
  "qdrant": "connected"
}
```

---

## 6. 数据预处理流程说明

系统将电子书摄入知识库的完整流程分为 5 个阶段：

```
文件上传 → 文件解析 → 图片 Caption → 文本分块 → 向量化 → 写入 Qdrant
```

### 6.1 文件解析

**PDF 解析**（`backend/core/parsers/pdf_parser.py`）

- 使用 PyMuPDF (fitz) 逐页提取文字
- 自动提取内嵌图片，过滤小于 100×100 像素的噪声图
- 使用 Pillow 将图片压缩至最大 1024×1024 像素
- 保留页码信息，写入 chunk payload 供前端展示

**EPUB 解析**（`backend/core/parsers/epub_parser.py`）

- 使用 ebooklib 读取 OPF 包结构
- BeautifulSoup4 解析各 HTML 章节，剥离标签提取纯文字
- 提取封面图和正文内嵌图片
- 保留章节标题作为文档元数据

**Markdown 解析**（`backend/core/parsers/markdown_parser.py`）

- 按 `##` / `###` 标题分段，每段作为独立文本块
- 正则提取 `![alt](path)` 图片引用，解析本地图片路径

### 6.2 图片 Caption 生成

图片不直接向量化，而是先由 Vision LLM 生成文字描述，再作为文本 chunk 处理：

```
图片文件
  → Vision LLM (doubao-seed-2-0-lite / Ollama llava)
  → 生成中文描述："这是一张展示神经网络结构的图表，包含输入层..."
  → 创建文本 chunk（type=image）
  → chunk.payload.image_path 保存图片磁盘路径（前端可请求 /images/{path} 展示原图）
```

Caption 提供商优先级：

1. **Doubao Vision**：质量最高，需要 `ARK_API_KEY`
2. **Ollama llava**：本地推理，无需 API Key，但需要先拉取模型
3. **降级（无 Caption）**：图片 chunk 使用占位文字，不影响文字检索功能

### 6.3 文本分块策略

使用 `RecursiveTextSplitter`，按以下优先级尝试切分点：

1. `\n\n`（段落边界）
2. `\n`（换行）
3. `。` / `！` / `？`（中文句子结束）
4. ` `（单词边界）
5. 字符级切分（兜底，避免超出限制）

默认参数：`chunk_size=512, chunk_overlap=64`（可通过 `CHUNK_SIZE` / `CHUNK_OVERLAP` 环境变量调整）

### 6.4 双路向量化

每个 chunk 生成两路向量：

| 类型 | 方法 | 存储字段 | 说明 |
|------|------|----------|------|
| Dense（稠密） | Embedding 模型（doubao / ollama） | Qdrant `dense` 命名向量 | 语义相似度，余弦距离 |
| Sparse（稀疏） | FastEmbed BM25 (`Qdrant/bm25`) | Qdrant `sparse` 命名向量 | 关键词精确匹配 |

> BM25 模型首次使用时自动下载（约 100 MB），需要网络连接。下载完成后缓存到本地，后续无需下载。

### 6.5 混合检索与融合（RRF）

查询时同时执行两路检索，再用倒数排名融合（RRF, Reciprocal Rank Fusion）合并结果：

```
用户查询文字
  ├─→ Dense Embedding → 语义检索（Top-K×2）
  └─→ BM25 Sparse → 关键词检索（Top-K×2）
       ↓
     RRF 融合排序
       ↓
     （可选）CrossEncoder Reranker 二次排序
       ↓
     返回最终 Top-K 结果
```

### 6.6 Chunk Payload 结构

每条 Qdrant 向量点的 payload 包含：

```json
{
  "doc_id": "d4e5f6a7-...",
  "kb_id": "3f8a1b2c-...",
  "chunk_id": "chunk-uuid",
  "text": "注意力机制允许模型...",
  "chunk_type": "text",
  "source_doc": "deep_learning.pdf",
  "page_num": 147,
  "image_path": null
}
```

图片 chunk 的 `image_path` 非空，指向 `storage/images/` 目录下的图片文件，前端通过 `/images/{path}` 路径访问。

### 6.7 文档状态流转

```
pending → processing → indexed
              ↓
            failed（可查看 error_message）
```

摄入失败后可以删除文档重新上传，或查看后端日志排查原因。

---

## 7. 多模型配置指南

### 7.1 豆包（Doubao / 火山引擎 Ark）

**适用场景**：推荐默认选择，支持中文优化，支持多模态（图文混排书籍）

**配置步骤**：

1. 前往 [火山引擎控制台](https://console.volcengine.com/ark) 注册并创建 API Key
2. 在 Ark 平台开通以下模型的访问权限：
   - `doubao-embedding-vision-251215`（Embedding）
   - `doubao-seed-2-0-lite-260428`（问答）
3. 在 `.env` 中配置：

```bash
ARK_API_KEY=your_ark_api_key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# Embedding 模型（文字 + 图片，统一走多模态接口）
DOUBAO_TEXT_EMBEDDING_MODEL=doubao-embedding-vision-251215
DOUBAO_VISION_EMBEDDING_MODEL=doubao-embedding-vision-251215

# 问答模型
DOUBAO_CHAT_MODEL=doubao-seed-2-0-lite-260428

# 图片 Caption 模型（同问答模型即可）
DOUBAO_VISION_CHAT_MODEL=doubao-seed-2-0-lite-260428
```

**创建知识库时的参数：**

```bash
python cli/main.py kb create \
  --name "豆包知识库" \
  --embedding-provider doubao \
  --model doubao-embedding-vision-251215 \
  --dim 2048 \
  --multimodal
```

**注意事项**：

- 豆包 vision embedding 使用 `/embeddings/multimodal` 端点（非标准 `/embeddings`），系统已内置适配，无需额外配置
- 批量 embedding 请求有时返回条数少于请求数，系统已内置自动降级为逐条请求的逻辑

### 7.2 DeepSeek

**适用场景**：高质量推理，适合学术和技术类问答；纯文字（不支持图片查询）

**配置步骤**：

1. 前往 [DeepSeek 开放平台](https://platform.deepseek.com) 获取 API Key
2. 在 `.env` 中配置：

```bash
DEEPSEEK_API_KEY=sk-your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

**使用 DeepSeek 作为问答模型：**

DeepSeek 仅支持作为问答 LLM，不支持作为 Embedding 提供商。创建知识库时仍需要使用 doubao 或 ollama 的 embedding 模型。在前端对话设置中，选择 LLM 提供商为 `deepseek` 即可。

```bash
# CLI 中指定 DeepSeek 作为问答模型
python cli/main.py chat --kb-id 3f8a1b2c-... --llm-provider deepseek
```

### 7.3 Ollama（本地模型）

**适用场景**：离线使用、数据隐私要求高、不想依赖云 API

**安装 Ollama：**

```bash
# macOS
curl -fsSL https://ollama.ai/install.sh | sh

# 或从官网下载：https://ollama.ai
```

**拉取所需模型：**

```bash
# 问答模型（按需选择一个）
ollama pull llama3           # 英文为主
ollama pull qwen2.5:7b       # 中文优化，推荐
ollama pull deepseek-r1:7b   # 推理增强

# Embedding 模型
ollama pull nomic-embed-text # 英文优化，768 维
ollama pull mxbai-embed-large # 更高质量，1024 维

# 多模态（图片 Caption）
ollama pull llava:13b
```

**配置 `.env`：**

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_CHAT_MODEL=qwen2.5:7b
OLLAMA_DEFAULT_EMBED_MODEL=nomic-embed-text
```

**创建使用 Ollama Embedding 的知识库：**

```bash
# nomic-embed-text 维度为 768
python cli/main.py kb create \
  --name "本地书库" \
  --embedding-provider ollama \
  --model nomic-embed-text \
  --dim 768

# mxbai-embed-large 维度为 1024
python cli/main.py kb create \
  --name "高质量本地书库" \
  --embedding-provider ollama \
  --model mxbai-embed-large \
  --dim 1024
```

**注意**：Ollama embedding 模型的向量维度必须与创建知识库时填写的 `--dim` 严格一致，否则查询时返回 409 错误。

### 7.4 vLLM（自托管高性能推理）

**适用场景**：GPU 服务器自托管，高并发生产部署

**启动 vLLM 服务：**

```bash
# 安装 vLLM
pip install vllm

# 启动服务（以 Qwen2.5-7B 为例）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8080 \
  --served-model-name qwen2.5-7b
```

**配置 `.env`：**

```bash
VLLM_BASE_URL=http://your-gpu-server:8080/v1
VLLM_API_KEY=EMPTY
```

**使用 vLLM 作为问答模型：**

创建会话时选择 `llm_provider=vllm`，`llm_model_id` 填写 vLLM 启动时的 `--served-model-name`（如 `qwen2.5-7b`）。

vLLM 目前不支持作为 Embedding 提供商，Embedding 仍需使用 doubao 或 ollama。

### 7.5 模型组合建议

| 场景 | Embedding 模型 | 问答模型 | 说明 |
|------|----------------|----------|------|
| 标准中文书籍 | doubao-embedding-vision-251215 | doubao-seed-2-0-lite | 默认推荐 |
| 图文混排书籍 | doubao-embedding-vision-251215（多模态） | doubao-seed-2-0-lite | 开启 `--multimodal` |
| 完全本地化 | ollama/nomic-embed-text | ollama/qwen2.5:7b | 零 API 费用，需 GPU |
| 高质量推理 | doubao-embedding-vision-251215 | deepseek-chat | Embedding 走豆包，问答走 DeepSeek |
| 生产高并发 | doubao-embedding-vision-251215 | vLLM/自托管模型 | Embedding 走豆包，LLM 自托管 |

---

## 8. 常见问题排查（FAQ）

### Q1：启动后端报错 `qdrant connection refused`

**原因**：Qdrant 未启动或端口不通。

**解决方案**：

```bash
# 确认 Qdrant 正在运行
docker ps | grep qdrant

# 若未运行，启动 Qdrant
make dev-qdrant

# 验证 Qdrant 可访问
curl http://localhost:6333/healthz
```

---

### Q2：上传文件后状态一直是 `pending`，不转 `processing`

**原因**：后台摄入任务未正常启动。

**解决方案**：

```bash
# 查看后端日志，寻找报错信息
make dev-backend
# 或查看 Docker 容器日志
docker compose logs -f backend
```

常见原因：
- Python 依赖缺失（重新执行 `make install`）
- ARK_API_KEY 未配置导致 embedding 调用失败
- Qdrant 服务中断

---

### Q3：摄入失败，状态变为 `failed`，错误信息为 `Doubao embedding API error`

**原因**：豆包 API Key 无效或模型未开通。

**解决方案**：

1. 确认 `.env` 中的 `ARK_API_KEY` 正确
2. 前往火山引擎控制台确认 `doubao-embedding-vision-251215` 模型已开通
3. 测试 API Key 是否有效：

```bash
curl https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"doubao-embedding-vision-251215","input":[{"type":"text","text":"测试"}]}'
```

---

### Q4：查询时返回 `409 Conflict` 错误

**原因**：查询使用的 embedding 模型与知识库创建时锁定的模型不一致（维度不匹配）。

**解决方案**：

系统严格校验 `(provider, model_id, vector_dim)` 三元组。如果需要更换 embedding 模型，必须：

1. 删除旧知识库：`python cli/main.py kb delete --kb-id xxx --yes`
2. 用新模型重新创建知识库
3. 重新摄入所有文档

---

### Q5：BM25 稀疏检索报错 `Failed to download Qdrant/bm25`

**原因**：首次使用 BM25 需要下载约 100 MB 的 ONNX 模型，网络不通。

**解决方案**：

```bash
# 手动下载 fastembed BM25 模型
python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"
```

如果网络受限，可以使用代理或手动下载后放置到 fastembed 缓存目录（通常为 `~/.cache/fastembed/`）。

---

### Q6：PDF 文件摄入后图片没有 Caption

**原因**：Vision LLM 未配置或调用失败。

**解决方案**：

1. 确认 `ARK_API_KEY` 配置正确，或 Ollama 正在运行且已拉取 llava 模型
2. 图片 Caption 失败不影响文字内容的摄入，文字部分仍会正常索引
3. 若需要多模态检索，修复 Vision LLM 配置后，清空知识库重新摄入：

```bash
python cli/main.py kb clear --kb-id xxx --yes
python cli/main.py ingest ./book.pdf --kb-id xxx
```

---

### Q7：检索结果质量差，找不到相关内容

**可能原因及解决方案**：

1. **chunk_size 设置不合理**：技术书籍推荐调整到 600-800，专业术语密集的场景适当增大

   ```bash
   # .env 中调整
   CHUNK_SIZE=700
   CHUNK_OVERLAP=100
   ```

2. **Reranker 未生效**：重排默认开启，但若 rerank 服务未启动（日志里有 `rerank_failed`）会自动回退，等于没开；检查 `RERANK_BASE_URL` 与 vLLM 服务

3. **Top-K 过小**：增大 `DEFAULT_TOP_K` 至 10-15

4. **embedding 模型不匹配**：中文书籍强烈建议使用豆包 embedding，英文书籍可以考虑 nomic-embed-text

---

### Q8：流式回答突然中断，前端无响应

**原因**：通常是 LLM 请求超时或网络中断。

**解决方案**：

1. 检查后端日志中是否有超时错误
2. 在 `.env` 中增大 LLM 超时配置（在 `llm_config` 中传递 `timeout` 参数）
3. 尝试缩短问题或减小 `top_k` 以减少上下文长度

---

### Q9：Docker 容器内无法拉取 Qdrant 镜像

**原因**：国内网络无法访问 Docker Hub。

**解决方案**：

项目已配置使用 DaoCloud 镜像源（`docker.m.daocloud.io`），`docker-compose.yml` 和 `docker-compose.dev.yml` 均已内置此配置，直接使用 `make dev-qdrant` 即可。

---

### Q10：EPUB 文件摄入后中文显示乱码

**原因**：EPUB 文件编码非 UTF-8。

**解决方案**：

通常 EPUB 文件本身是标准 HTML 格式（UTF-8），如果遇到乱码，可以先用 Calibre 等工具转换为 UTF-8 编码的 EPUB 或 PDF 后再上传。

---

## 9. 生产部署建议

### 9.1 使用 Docker Compose 生产模式

项目已提供 `docker-compose.yml`，包含 Qdrant + PostgreSQL + 后端 + 前端四个服务：

```bash
# 1. 配置生产环境变量
cp .env.example .env
# 编辑 .env，填写所有必要的 API Key 和生产配置

# 2. 构建并启动
make up
# 等价于：docker compose up -d

# 3. 初始化数据库（首次部署）
docker compose exec backend alembic upgrade head

# 4. 查看后端日志
make logs-backend

# 5. 停止服务
make down
```

### 9.2 切换为 PostgreSQL 数据库

生产环境强烈建议从 SQLite 切换到 PostgreSQL：

```bash
# docker-compose.yml 中后端已默认连接 PostgreSQL
DATABASE_URL=postgresql+asyncpg://bookrag:bookrag@postgres:5432/bookrag
```

如需使用外部 PostgreSQL 实例：

```bash
DATABASE_URL=postgresql+asyncpg://username:password@your-db-host:5432/bookrag
```

### 9.3 Nginx 反向代理配置

建议在前端和后端前部署 Nginx：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /app/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # SSE 流式接口（必须关闭缓冲）
    location /api/v1/sessions/ {
        proxy_pass http://backend:8000;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding on;
    }

    # 图片静态文件
    location /images/ {
        proxy_pass http://backend:8000;
    }
}
```

> **重要**：SSE 流式接口必须配置 `proxy_buffering off`，否则 token 会被 Nginx 缓冲，导致用户看不到流式输出。

### 9.4 存储持久化

生产部署必须挂载持久化卷：

```yaml
# docker-compose.yml 中已配置
volumes:
  - ./storage:/app/storage      # 上传的文件和图片
  - qdrant_data:/qdrant/storage  # Qdrant 向量数据
  - postgres_data:/var/lib/postgresql/data  # PostgreSQL 数据
```

**备份策略建议**：

```bash
# 每日备份 PostgreSQL
docker compose exec postgres pg_dump -U bookrag bookrag > backup_$(date +%Y%m%d).sql

# 备份 storage 目录（文档文件）
tar -czf storage_backup_$(date +%Y%m%d).tar.gz ./storage/

# Qdrant 可通过快照 API 备份
curl -X POST http://localhost:6333/collections/{collection_name}/snapshots
```

### 9.5 Qdrant 生产配置

为 Qdrant 开启访问控制：

```yaml
# docker-compose.yml 环境变量中添加
environment:
  QDRANT__SERVICE__API_KEY: your_qdrant_api_key
```

同步更新 `.env`：

```bash
QDRANT_API_KEY=your_qdrant_api_key
```

### 9.6 CORS 配置

生产环境应限制 CORS 来源为实际域名：

```bash
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
```

### 9.7 性能调优

**后端 Worker 数量**（高并发场景）：

```bash
# 默认单 worker，多 CPU 服务器可增加
uvicorn main:app --workers 4 --port 8000
```

**Qdrant 内存配置**（大规模向量库）：

当向量数量超过 100 万时，建议为 Qdrant 容器配置足够内存（至少 4 GB），并在 Qdrant 配置中启用向量量化以节省内存：

```yaml
# Qdrant 创建集合时可指定量化（需要代码层支持）
# 参考 Qdrant 官方文档：https://qdrant.tech/documentation/guides/quantization/
```

**分块参数调整**（内容质量优化）：

| 书籍类型 | 推荐 CHUNK_SIZE | 推荐 CHUNK_OVERLAP |
|----------|----------------|-------------------|
| 小说/散文 | 300-400 | 50 |
| 技术教材 | 512-700 | 64-100 |
| 学术论文 | 600-800 | 100 |
| 代码文档 | 400-600 | 50 |

### 9.8 监控与告警

建议监控以下指标：

- **后端健康**：定期 `GET /api/v1/health`，Qdrant 断连时返回 503
- **摄入队列积压**：监控 `pending` 状态文档数量，超过阈值告警
- **向量库大小**：监控 Qdrant 磁盘使用量
- **API 响应延迟**：重点关注 `/chat/stream` 首 token 延迟

```bash
# 简单健康检查脚本示例
#!/bin/bash
HEALTH=$(curl -s http://localhost:8000/api/v1/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
if [ "$HEALTH" != "ok" ]; then
  echo "BookRAG backend unhealthy!"
  # 发送告警...
fi
```

---

*如有疑问或发现文档错误，请检查 `docs/ARCHITECTURE.md` 获取更多系统设计细节，或查看 `backend/` 源码中的内联注释。*
