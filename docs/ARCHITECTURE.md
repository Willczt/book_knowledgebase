# 系统架构

## 整体架构

```mermaid
graph TB
    subgraph 客户端层
        WEB[React 前端<br/>Tailwind + Zustand]
        CLI[CLI 终端工具<br/>Typer + Rich]
    end

    subgraph API 层
        FASTAPI[FastAPI<br/>REST + SSE 流式]
        CORS[CORS / Proxy]
    end

    subgraph RAG 核心
        INGEST[摄入流水线<br/>parse → chunk → caption → embed → index]
        RETRIEVAL[混合检索<br/>Dense + Sparse BM25 + RRF]
        PIPELINE[RAG Pipeline<br/>检索 → Prompt → LLM 流式生成]
    end

    subgraph 存储层
        QDRANT[(Qdrant 向量库<br/>密集 + 稀疏双向量)]
        DB[(SQLite / PostgreSQL<br/>元数据 + 会话历史)]
        FS[文件系统<br/>原文件 + 图片缓存]
    end

    subgraph 模型层
        subgraph Embedding
            EMB_DOUBAO[豆包 Embedding<br/>doubao-embedding-*]
            EMB_OLLAMA[Ollama Embedding]
            EMB_VLLM[vLLM Embedding]
        end
        subgraph LLM
            LLM_DOUBAO[豆包 Chat<br/>doubao-seed-*]
            LLM_DEEPSEEK[DeepSeek<br/>deepseek-chat/coder]
            LLM_OLLAMA[Ollama<br/>llama3 / llava]
            LLM_VLLM[vLLM]
        end
    end

    subgraph 微调模块（可选）
        FT_DATA[数据准备<br/>QA 对生成]
        FT_TRAIN[Unsloth LoRA 训练]
        FT_EXPORT[导出 GGUF → Ollama]
    end

    WEB --> CORS --> FASTAPI
    CLI --> FASTAPI
    FASTAPI --> INGEST
    FASTAPI --> PIPELINE
    INGEST --> QDRANT
    INGEST --> DB
    INGEST --> FS
    RETRIEVAL --> QDRANT
    PIPELINE --> RETRIEVAL
    PIPELINE --> LLM_DOUBAO
    PIPELINE --> LLM_DEEPSEEK
    PIPELINE --> LLM_OLLAMA
    PIPELINE --> LLM_VLLM
    INGEST --> EMB_DOUBAO
    INGEST --> EMB_OLLAMA
    INGEST --> EMB_VLLM
    FT_DATA --> FT_TRAIN --> FT_EXPORT --> LLM_OLLAMA
```

## 数据摄入流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant API as FastAPI
    participant PARSER as 解析器
    participant CAPTIONER as 图片 Captioner
    participant EMBED as Embedding 模型
    participant QDRANT as Qdrant

    U->>API: 上传文件 (PDF/EPUB/MD)
    API->>API: 创建 Document 记录 (status=pending)
    API-->>U: 返回 doc_id
    API->>PARSER: 后台任务：解析文件
    PARSER->>PARSER: 提取文字块 + 图片
    loop 图片
        PARSER->>CAPTIONER: LLM Vision Caption
        CAPTIONER-->>PARSER: 中文描述文字
    end
    PARSER->>PARSER: RecursiveTextSplitter 分块
    loop 每个 chunk
        PARSER->>EMBED: Dense Embedding
        PARSER->>EMBED: BM25 Sparse Embedding (FastEmbed)
        PARSER->>QDRANT: upsert (dense + sparse + payload)
    end
    API->>API: 更新 Document status=indexed
```

## 检索与 RAG 流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant API as FastAPI
    participant EMBED as Embedding 模型
    participant QDRANT as Qdrant
    participant RERANKER as CrossEncoder Reranker
    participant LLM as LLM

    U->>API: 发送问题（+ 可选图片）
    API->>API: 校验 KB Embedding 模型一致性
    API->>EMBED: Query Dense Embedding
    API->>QDRANT: Prefetch Dense (top-50)
    API->>QDRANT: Prefetch Sparse BM25 (top-50)
    QDRANT->>API: RRF Fusion (top-20)
    API->>RERANKER: CrossEncoder 重排序 (top-5)
    API->>LLM: 组装 Prompt，流式生成
    loop SSE 流式
        LLM-->>API: token
        API-->>U: data: {"token": "..."}
    end
    API-->>U: data: {"done": true, "sources": [...]}
```

## Embedding 模型锁定机制

```mermaid
graph LR
    CREATE[创建知识库] --> LOCK[锁定: provider + model_id + vector_dim]
    LOCK --> STORE[(KnowledgeBase 表)]
    QUERY[检索/摄入] --> VALIDATE{校验三元组匹配?}
    VALIDATE -- Yes --> OK[继续]
    VALIDATE -- No --> ERROR[409 Conflict]
    STORE --> VALIDATE
```

> **设计原则**: 向量空间由 Embedding 模型决定。若在摄入阶段和检索阶段使用了不同模型，
> 向量无法对齐，检索结果将完全错误（静默失败）。因此系统强制要求 Embedding 模型不可变。
