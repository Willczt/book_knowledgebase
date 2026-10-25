# 数据预处理指南

本文档详细介绍如何将电子书（PDF、EPUB、Markdown、TXT）摄入知识库。

---

## 支持的文件格式

| 格式 | 特点 | 图片处理 |
|------|------|----------|
| PDF（非扫描） | 文字可选取的 PDF | 提取内嵌图片，LLM Caption |
| EPUB | HTML 结构的电子书 | 提取 OPF 资源图片 |
| Markdown | 结构化文档 | 解析 `![alt](path)` 引用 |
| TXT | 网络小说纯文本（UTF-8 / GBK / GB18030 / Big5） | 无 |

---

## 摄入流水线详解

### 1. 文件解析

**PDF** (`backend/core/parsers/pdf_parser.py`)
- 使用 PyMuPDF (fitz) 逐页提取文字和图片
- 过滤小尺寸图片（默认 100×100 px 以下）
- 用 Pillow 将图片 resize 到最大 1024×1024

**EPUB** (`backend/core/parsers/epub_parser.py`)
- 使用 ebooklib 读取 OPF 包
- BeautifulSoup4 解析 HTML 章节，提取纯文字（只取叶子块级节点，避免容器 div 与子 p 重复输出）
- 提取封面图和内嵌图片；同一张图只解码一次（按 src 缓存），尺寸达标时沿用原始编码不重编码

**Markdown** (`backend/core/parsers/markdown_parser.py`)
- 按 `##`/`###` 标题分段
- 正则提取 `![alt](path)` 图片引用

**TXT** (`backend/core/parsers/txt_parser.py`)
- 自动识别编码（UTF-8 / UTF-16 BOM / GB18030 / Big5），网络小说常见的 GBK 文件可直接上传
- 按「第N章 / 卷N / 序 / 楔子 / 番外 / 终章」等标题行切分章节，标题写入 chunk 的 `chapter` 元数据
- 清理全角空格缩进与多余空行；无章节标记的超大文件按 2 万字切片

### 2. 图片 Caption

图片不直接向量化，而是通过 Vision LLM 生成文字描述后作为文本 chunk 处理：

```
图片 → Vision LLM (doubao/llava) → 中文描述 → 文字 chunk
chunk.payload.image_path = "图片磁盘路径"  # 前端可展示原图
```

**Caption 提供商优先级**：
1. Doubao Vision（`doubao-vision-*`）
2. Ollama llava（本地）
3. 降级：使用占位文字（图片功能不可用）

### 3. 文本分块

使用 `RecursiveTextSplitter`，按优先级尝试分隔符：
1. `\n\n`（段落）
2. `\n`（换行）
3. `。`/`！`/`？`（中文句号）
4. ` `（空格）
5. 字符级切分（兜底）

默认参数：`chunk_size=1000, chunk_overlap=200`

可通过环境变量调整：
```bash
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

### 4. 向量化

每个 chunk 生成两路向量：

**Dense（稠密）**：由配置的 Embedding 模型生成，存储到 Qdrant `"dense"` 命名向量
**Sparse（稀疏）**：FastEmbed BM25（`Qdrant/bm25`），进程内 ONNX，无需外部服务

⚠️ **重要**：Dense Embedding 模型在知识库创建时锁定，后续所有摄入和检索必须使用相同模型。

---

## 通过界面上传

1. 在左侧侧边栏选择目标知识库（展开它）
2. 拖拽或点击上传区域，选择文件（支持多选）
3. 观察文件状态：
   - 灰色点：等待处理
   - 黄色脉冲点：处理中
   - 绿色点：已索引
   - 红色点：处理失败

---

## 通过 CLI 摄入

```bash
# 摄入单个文件
book-rag ingest --kb-id <kb-id> path/to/book.pdf

# 摄入整个目录
book-rag ingest --kb-id <kb-id> path/to/books/

# 查看状态
book-rag doc list --kb-id <kb-id>
```

---

## 清空知识库

```bash
# 通过 CLI
book-rag kb clear <kb-id>

# 通过界面
# 在 KB 旁边点击「刷新」图标（清空向量）
# 或点击「删除」图标（删除整个知识库）
```

清空操作会删除 Qdrant 中的所有向量并重建空集合，但 SQLite 中的文档记录也会清空，需重新上传。

---

## 常见问题

### Q: 摄入失败，显示红色点
- 检查后端日志：`make dev-backend` 看 Python 输出
- 常见原因：PDF 加密、文件编码问题、Qdrant 未启动

### Q: 检索结果质量差
- 确认 chunk_size 设置合理（对于技术书籍可尝试 600-800）
- 图片 Caption 质量影响多模态检索：确保 Vision LLM 配置正确
- 考虑开启 CrossEncoder Reranker（`ENABLE_RERANKER=true`）

### Q: 上传图文混排 PDF 后图片没有 Caption
- 检查环境变量是否设置了 `ARK_API_KEY`（Doubao Vision）或 Ollama 是否运行了 llava 模型
- 降级后图片 chunk 会有占位文字，不影响文字检索
