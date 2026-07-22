# 生产级 RAG 系统技术面试 Q&A

> 本文档面向电子书知识库 RAG 系统，针对国内一线互联网公司（字节/阿里/腾讯/美团）高级/资深工程师岗位撰写。每道题均包含**答题思路、核心答案、代码/架构示例、追问预判**四个维度。

---

## 第一章：RAG 系统整体设计

---

### Q1-1：为什么选择 RAG 而不是 Fine-tuning？两者的适用场景是什么？

**答题思路**

先给出结论性判断，再从"知识更新频率、训练成本、幻觉控制、可解释性"四个维度对比，最后说明本系统的具体场景决策。

**核心答案**

RAG 和 Fine-tuning 解决的是两类不同问题。Fine-tuning 主要改变模型的**行为模式**（语气、格式、推理风格、领域术语理解），而 RAG 是给模型**注入运行时知识**（具体事实、文档内容、最新数据）。

本系统选择 RAG 而非 Fine-tuning 的核心原因：

1. **知识动态性**：用户持续上传新书，知识库内容每天变化。Fine-tuning 一旦完成就把知识"烘焙"进了权重，新内容需要重新训练，成本极高。RAG 只需将新文档入库（解析→分块→embedding→写入 Qdrant）即可立即可用。

2. **幻觉可控性**：RAG 强制模型基于检索到的上下文作答（Prompt 中明确写"只基于提供的参考资料回答"），可在 sources 字段返回来源溯源，方便验证。Fine-tuning 则是将知识内化进权重，出现幻觉时难以定位。

3. **训练数据成本**：高质量的领域 SFT 数据构建成本极高，且电子书内容涉及版权，很难大规模标注。

4. **可解释性**：每次回答都附带 `sources: [{source_doc, page_num, text_snippet}]`，企业合规场景下可明确追溯。

**适用场景对比**

| 维度 | RAG 适用 | Fine-tuning 适用 |
|------|---------|----------------|
| 知识更新频率 | 高（日更） | 低（年更） |
| 知识总量 | 超出 context window | 可压缩入少量样本 |
| 回答格式/风格 | 不敏感 | 需要特定输出格式/语气 |
| 推理能力增强 | 不适用 | 适用（数学、代码推理） |
| 成本 | 低（只需 embedding API） | 高（GPU + 数据标注） |
| 幻觉控制 | 强（有 grounding） | 弱 |

**本系统的混合策略**：系统同时提供了微调模块（`finetuning/`），用于将 Fine-tuning 作为**第二阶段增强**——用知识库中的 QA 对训练一个对书籍领域有更好理解的本地模型（Llama3/Qwen），但核心知识检索仍走 RAG 路径。

**追问预判**

- "如果同一本书的内容已经在预训练数据里了，RAG 还有价值吗？" — 有，预训练的知识是泛化的，RAG 检索的是书中的精确原文和章节位置，精确度更高。
- "RAG 系统的延迟比直接 LLM 高，怎么优化？" — 参见第七章性能优化。

---

### Q1-2：描述整个检索增强生成的完整数据流

**答题思路**

分两个阶段讲：**离线索引阶段**（文档入库）和**在线检索阶段**（用户问答），用具体的函数调用链串联。

**核心答案**

**离线索引阶段（Indexing Pipeline）**

```
用户上传文件 (POST /knowledge-bases/{kb_id}/documents)
    │
    ├── API 层：文件写入 storage/docs/{kb_id}/{doc_id}.pdf
    ├── SQLite 写入 Document 记录 (status=pending)
    └── asyncio.create_task(run_ingest_task(...))  ← 非阻塞后台任务
            │
            ├── [1/5] Parser 解析文档
            │       PDFParser(PyMuPDF) → Iterator[ParsedChunk]
            │       每个 ParsedChunk 含 text + images[] + page_num
            │
            ├── [2/5] 分块 + 图片 Caption
            │       RecursiveTextSplitter(chunk_size=512, overlap=64)
            │       ImageCaptioner → Vision LLM(doubao-seed-2-0) → 中文描述
            │       → ChunkPoint(text, chunk_type="text"|"image_caption")
            │
            ├── [3/5] Dense Embedding
            │       DoubaoEmbedding.embed_documents(batch_size=16)
            │       → 2048-dim float32 向量
            │
            ├── [4/5] Sparse BM25 Embedding
            │       FastEmbed SparseTextEmbedding("Qdrant/bm25")
            │       → sparse indices + values（ONNX，进程内运行）
            │
            └── [5/5] 写入 Qdrant
                    upsert_points(batch=64)，每个点含 dense + sparse 双向量
                    SQLite 更新 Document.status = "indexed"
```

**在线检索阶段（Query Pipeline）**

```
用户发送 POST /sessions/{session_id}/chat/stream
    │
    ├── 验证 Embedding 锁定（KB 存储的 provider/model_id/dim 必须精确匹配）
    ├── 加载历史对话（最近 10 轮）
    │
    └── RAGPipeline.stream_chat()
            │
            ├── [检索] HybridRetriever.build_query()
            │       ├── DoubaoEmbedding.embed_query(text, image?) → dense_vec
            │       └── FastEmbed BM25.embed([text]) → sparse_indices, sparse_values
            │
            ├── [向量检索] QdrantVectorStore.hybrid_search()
            │       Prefetch(dense, limit=20) + Prefetch(sparse, limit=20)
            │       → FusionQuery(Fusion.RRF) → top_k=8 个结果
            │
            ├── [可选] CrossEncoderReranker.rerank()
            │       FlashRank ms-marco-MiniLM，纯 ONNX，本地运行
            │
            ├── [生成] 构建 Prompt
            │       System Prompt + 检索到的 context + 历史消息 + 用户问题
            │
            └── [流式输出] LLM.stream_chat() → AsyncIterator[token]
                    yield {"token": "..."} × N
                    yield {"done": True, "sources": [...]}
                    StreamingResponse(media_type="text/event-stream")
```

**追问预判**

- "Embedding 锁定是什么机制，为什么必要？" — 参见 Q2-6。
- "如果图片很多，Caption 很慢怎么办？" — 索引是异步后台任务，不阻塞 API 响应；可以配置 `caption_provider=null` 跳过。

---

### Q1-3：如何保证检索结果的质量？（多个层次）

**答题思路**

分四层讲：检索前（数据质量）、检索中（算法）、检索后（重排序）、生成中（Prompt 设计）。

**核心答案**

**第一层：数据质量保障（Garbage In, Garbage Out）**

- **分块粒度控制**：`chunk_size=512` 字符，`overlap=64` 字符。块太大会引入噪声，块太小丢失上下文。512 字符约合 200-300 汉字，对应一个自然段落，是实验验证过的平衡点。
- **最小内容过滤**：PDF 解析时过滤 `len(text) < 20` 的碎片块（页码、页眉等），图片过滤 `size < 50x50` 的图标。
- **语义完整性**：RecursiveTextSplitter 优先按 `\n\n`（段落）→ `\n`（行）→ `。`（句子）分割，尽量在语义边界切分，而不是硬截断。
- **图片语义化**：图片转换为中文 Caption，Caption 进入文本索引，弥补纯文本检索对图表的盲点。

**第二层：混合检索（算法层面）**

- **Dense（语义）+ Sparse（词汇）互补**：Dense Embedding 擅长语义相似但用词不同的检索（同义词、换个说法），Sparse BM25 擅长精确词汇匹配（专有名词、型号、人名）。两者短板互补。
- **RRF 融合**：公平地融合两路结果，无需手工调权重，对两路排序结果都有贡献的文档获得更高综合分。

**第三层：二阶段重排序（Reranker）**

- CrossEncoder（FlashRank ms-marco-MiniLM）对 query 和每个候选 chunk 的**拼接文本**做精细化打分，比 Bi-encoder 更准确但更慢。实现上放在可选路径（`use_reranker=true`）。
- 设计上 prefetch 取 `limit=20`，rerank 后返回 `top_k=8`，漏斗式收窄。

**第四层：Prompt Engineering**

```
System Prompt：
  "只基于提供的参考资料回答，不要凭空捏造信息。
   如果参考资料中没有相关信息，请明确告知用户。"
```

强制 grounding，减少模型凭训练记忆发挥的空间。同时每个引用都标注 `来源：XXX 第 N 页`，便于用户验证。

**追问预判**

- "score_threshold 怎么设？" — 要依据具体模型和数据分布实验决定。本系统默认 `score_threshold=0.0`（不过滤），RRF 的分数不是概率，不适合直接阈值过滤，可以用 reranker 分数来过滤。

---

### Q1-4：RAG 系统的主要挑战和你的解决方案

**答题思路**

列举工程落地时遇到的真实挑战，每个挑战配对应的解决方案和取舍。

**核心答案**

| 挑战 | 现象 | 解决方案 |
|------|------|---------|
| 检索召回不足 | 用户问"这本书讲了什么"，检索到碎片化 chunk | 混合检索（Dense+Sparse）+ RRF，扩大 prefetch_limit |
| 向量空间污染 | 换了 Embedding 模型但旧向量还在 | Embedding 锁定机制，409 拦截错误请求 |
| 图表信息丢失 | PDF 里的图表用文字无法检索到 | Vision LLM Caption → 文本 chunk，image_path 存入 payload |
| 长文档处理 | 单文档超过 context window | 分块策略 + top-k 检索（只取最相关的几块） |
| 索引耗时长 | 上传后用户长时间等待 | asyncio.create_task 后台处理，pending→processing→indexed 状态轮询 |
| 中英混排分词 | 中文"。"和英文"." 分词规则不同 | RecursiveTextSplitter 同时支持中英文分隔符 `["\n\n", "\n", "。", ".", " "]` |
| API 限速/失败 | Embedding API 偶发 429/5xx | tenacity 重试（3 次，指数退避 1-10s） |
| 多租户向量隔离 | 不同知识库向量混用 | 每个 KB 独立 Qdrant collection，查询时 doc_ids_filter 进一步隔离 |

**追问预判**

- "如果用户上传了重复文档怎么办？" — 可以在上传时做文件 hash 去重，或者在 Qdrant 中按 doc_id 删除旧向量后重新入库（幂等 upsert）。

---

## 第二章：混合检索与向量化

---

### Q2-1：详细解释 Hybrid Search（Dense + Sparse + RRF）的原理和优势

**答题思路**

先讲单独 Dense 和 Sparse 各自的局限性，再讲两者结合的必要性，最后讲 RRF 的具体实现。

**核心答案**

**Dense Retrieval 的短板**

Dense Embedding 将文本映射到连续向量空间，通过余弦相似度检索。它对**语义相似但用词不同**的场景很强，但对**精确词汇匹配**（专有名词、缩写、型号、人名、罕见词）表现差——因为这些词在训练语料中出现少，向量表示不准确，而且向量空间里语义相近的词会聚在一起，导致精确词被"稀释"。

**Sparse Retrieval（BM25）的短板**

BM25 只关注词频和逆文档频率，对**同义词、换说法**完全无能为力。"机器学习"和"ML"对 BM25 来说是两个完全不同的词。

**两者结合的互补性**

```
用户 query："解释一下 HNSW 索引的工作原理"

Dense 结果：          Sparse(BM25) 结果：
1. 向量索引算法概述    1. HNSW 数据结构详解（精确词匹配）
2. ANN 近似最近邻     2. 图索引构建流程
3. 索引性能对比       3. HNSW 查询复杂度分析

RRF 融合后：
1. HNSW 数据结构详解  ← sparse 第1 + dense 贡献分
2. 向量索引算法概述   ← dense 第1
3. 图索引构建流程     ← sparse 第2
```

**在本系统中的实现**

```python
# qdrant_store.py - 单次 query_points 调用完成两路检索和融合
results = await self.client.query_points(
    collection_name=collection_name,
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=20,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # Qdrant 原生支持 RRF
    limit=top_k,
)
```

Qdrant 的 Prefetch + FusionQuery 机制在**一次网络请求**中完成了原本需要三次请求（dense 查询 + sparse 查询 + 客户端融合）的工作，大幅降低延迟。

**追问预判**

- "Qdrant 原生 RRF 和自己在客户端做 RRF 有什么区别？" — 原生在服务端做，省去两次网络往返，且服务端可以更精细地利用索引结构优化。
- "能不能用 DPR（Dense Passage Retrieval）替代 BM25？" — 可以，但 DPR 需要双塔结构和专门的对比学习训练，成本高。BM25 是零成本的强基线，在召回率上往往不比精调的 DPR 差太多。

---

### Q2-2：BM25 算法原理，与 TF-IDF 的区别，在 RAG 中的价值

**答题思路**

先解释 TF-IDF，再讲 BM25 的两个关键改进（词频饱和 + 文档长度归一化），最后说明在 RAG 中的实际价值。

**核心答案**

**TF-IDF 回顾**

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
TF(t, d)  = 词 t 在文档 d 中出现次数 / 文档 d 总词数
IDF(t)    = log(文档总数 / 包含词 t 的文档数)
```

TF-IDF 的问题：词频无上限（出现 100 次的权重是 10 次的 10 倍，不合理）；未考虑文档长度（长文档天然词频高）。

**BM25 的改进**

```
BM25(t, d) = IDF(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]

k1 ∈ [1.2, 2.0]：词频饱和参数（k1 越大，高频词权重越高）
b ∈ [0, 1]：文档长度归一化参数（b=0 不归一化，b=1 完全归一化）
|d|：文档长度，avgdl：平均文档长度
```

BM25 相比 TF-IDF 的核心改进：
1. **词频饱和**：分子 `TF × (k1+1)` 在 TF 很大时趋近于 `k1+1`（常数），避免词频无限增益。
2. **文档长度归一化**：分母中的 `b × |d|/avgdl` 惩罚长文档，使短文档中出现同样词频的词得分更高。

**在 RAG 中的价值**

在本系统中，BM25 通过 FastEmbed 的 ONNX 实现（`Qdrant/bm25` 模型）以**稀疏向量**形式存储在 Qdrant 中。具体来说：

- 词汇表中每个 token 对应一个维度 index
- 每个 chunk 存储其 BM25 权重的稀疏表示（只存非零位置）
- 查询时计算 query token 与 chunk 中重叠 token 的 BM25 分

真实价值案例：用户搜索"《深度学习》第五章 卷积网络"，BM25 能直接精确匹配"第五章"、"卷积网络"这样的结构化词汇；Dense Embedding 可能因为这些词在向量空间中不够独特而检索偏离。

**追问预判**

- "FastEmbed 的 BM25 实现是真正的 BM25 吗？" — 是的，Qdrant 实现了 BM25+ 变体，用 ONNX tokenizer 做分词然后计算稀疏权重，在进程内运行无需外部服务。

---

### Q2-3：Dense Embedding 的原理，余弦相似度 vs 内积距离

**答题思路**

先讲 Embedding 的本质（语义压缩），再讲两种距离度量的数学含义和适用场景，最后说明本系统的选择。

**核心答案**

**Dense Embedding 本质**

Transformer Encoder 将文本序列编码后，通过池化（通常是 [CLS] token 或平均池化）得到固定维度向量。这个向量是文本语义的**稠密压缩表示**，相似语义的文本在向量空间中靠近。

本系统使用 `doubao-embedding-vision-251215`，输出 2048 维向量，支持文本和图片的统一向量空间（多模态）。

**余弦相似度 vs 内积距离**

```
余弦相似度：cos(A, B) = A·B / (|A| × |B|)
内积（点积）：IP(A, B) = A·B = Σ(Ai × Bi)
```

核心区别：

| 维度 | 余弦相似度 | 内积 |
|------|-----------|------|
| 归一化 | 是（除以模长） | 否 |
| 向量模长影响 | 无影响 | 有影响（长向量得分高） |
| 适用场景 | 大多数语义检索 | 向量已归一化 / 需要量级信息 |
| 计算成本 | 略高（需额外除法） | 低 |

**本系统的选择**

```python
# qdrant_store.py
VectorParams(size=vector_dim, distance=Distance.COSINE)
```

选择余弦相似度的原因：
1. Doubao Embedding 的向量**未强制归一化**，直接用内积可能受向量模长影响，导致较长文本（模长通常更大）被系统性偏好。
2. 余弦相似度关注**方向**（语义方向），与向量的绝对大小无关，更纯粹地反映语义相似性。
3. Qdrant 对余弦距离有 HNSW 索引优化，查询效率并不差。

**注意**：如果使用已做 L2 归一化的 Embedding 模型（如 OpenAI text-embedding-3），余弦等价于内积，此时两者没有区别，可以用更快的 IP。

**追问预判**

- "2048 维是不是太高了，有没有降维的必要？" — 2048 维已经是较高维度，PCA 降维有损失，不推荐。Qdrant 的 HNSW 索引能很好处理 2048 维，实际查询延迟通常在 10ms 以内。
- "什么时候选 L2 距离（欧氏距离）？" — L2 在图像检索、人脸识别中常用（关注绝对位置），语义文本检索通常用余弦。

---

### Q2-4：RRF（Reciprocal Rank Fusion）算法原理，为什么优于简单加权融合

**答题思路**

先给出 RRF 公式，再解释其核心思想，最后对比加权融合的缺点。

**核心答案**

**RRF 公式**

```
RRF_score(d) = Σ_r [ 1 / (k + rank_r(d)) ]

d：候选文档
r：排序系统（本系统中 r ∈ {dense, sparse}）
rank_r(d)：文档 d 在系统 r 中的排名（从 1 开始）
k：平滑常数，通常取 60（防止排名第 1 的文档分值过高）
```

**直观理解**

| dense 排名 | sparse 排名 | RRF 得分 |
|-----------|------------|---------|
| 1 | 1 | 1/(60+1) + 1/(60+1) ≈ 0.0328 |
| 1 | 不存在 | 1/(60+1) ≈ 0.0164 |
| 5 | 3 | 1/65 + 1/63 ≈ 0.0312 |
| 1 | 20 | 1/61 + 1/80 ≈ 0.0289 |

关键特性：**同时出现在两个排名中的文档会获得双倍加分**。这正是混合检索的期望行为——两种方法都认为好的文档应该排在前面。

**为什么优于简单加权融合**

简单加权融合：`score = α × dense_score + (1-α) × sparse_score`

问题：
1. **分数不可比**：Dense 余弦相似度在 [-1, 1]，BM25 可能在 [0, 10]，直接加权毫无意义。
2. **需要调参**：α 的最优值依赖数据分布，换个领域就要重新调。
3. **分数分布差异**：不同 Embedding 模型产生的分数分布（均值、方差）不同，加权会被分布宽的一方主导。

RRF 只使用**排名**而不使用分数绝对值，天然解决了分数不可比问题，且无需调参（k=60 是经验性默认值，在多个 benchmark 上均有良好表现）。

**追问预判**

- "k=60 是怎么来的？能调吗？" — 来自原始 RRF 论文的实验结论，对大多数场景都合适。k 越小，第一名的权重越大；k 越大，排名差距的影响越小，结果越平滑。
- "RRF 有没有缺点？" — 有：它丢失了分数的绝对大小信息。如果 dense 检索到一个非常确定的结果（余弦相似度 0.98）和一个不太确定的（0.6），RRF 无法区分，只看排名。可以用 reranker 做补偿。

---

### Q2-5：Embedding 向量维度对检索性能的影响

**答题思路**

从表达能力、存储成本、查询延迟三个维度分析，结合本系统实际。

**核心答案**

**维度与表达能力**

高维向量可以编码更多语义信息，表示更细粒度的语义区别。2048 维相比 768 维（BERT 默认）可以存储更多的"语义维度"，有利于检索精度。但在实践中，收益是边际递减的——从 512 到 1024 维提升明显，从 1024 到 2048 提升有限。

**存储成本**

```
每个向量存储大小 = dim × 4 bytes (float32)

768 维：3KB/向量
1024 维：4KB/向量
2048 维：8KB/向量

本系统 10 万个 chunk → 2048 维 dense：约 800MB
```

Qdrant 默认使用 float32，支持降低到 float16 或 int8（量化）以节省空间，但精度有损失。

**查询延迟（HNSW 索引）**

HNSW 的复杂度约为 O(log N)，与维度的关系相对不那么敏感（维度从 768 到 2048，查询延迟增加约 30-50%，而不是成倍增加）。真正影响查询延迟的是**集合大小 N** 和 **HNSW 参数（ef_construction, m）**。

**本系统设计决策**

使用 2048 维是随 Doubao Embedding 模型固定的，无法选择。如果需要权衡，可以考虑：
- 对向量做 PCA 降维到 512 维（有精度损失，约 3-5%）
- 使用 Qdrant 的向量量化（Scalar Quantization 或 Product Quantization）

**追问预判**

- "如果换一个 1024 维的模型，已有数据怎么迁移？" — 必须重新 embedding 全部数据（向量空间完全不同），这就是为什么需要 Embedding 锁定机制。

---

### Q2-6：为什么需要 Embedding 模型锁定机制？不锁定会出现什么问题？

**答题思路**

这是本系统的一个关键工程设计决策，要讲清楚问题根源、具体场景、解决方案，以及 409 状态码的设计理由。

**核心答案**

**不锁定会出现什么问题**

向量数据库存储的是特定 Embedding 模型产生的向量。**不同模型的向量空间是完全不可比的**——即使维度相同（都是 2048 维），模型 A 的 "苹果" 向量和模型 B 的 "苹果" 向量在彼此的空间里是随机方向的，没有语义意义。

具体场景：
```
时间线：
T1：用 doubao-embedding-vision-251215（2048 维）对书籍 A 入库
T2：管理员更换为 text-embedding-3-large（3072 维）——维度直接不匹配，Qdrant 报错
T3：管理员更换为另一个 2048 维模型——维度匹配，但向量空间不同！

T3 的情况是静默错误（Silent Corruption）：
- 系统不报错
- 老向量（T1 入库）和新查询向量在不同空间
- 检索结果随机，但没有任何报错
- 用户看到的是莫名其妙的检索结果，很难 debug
```

**本系统的锁定机制**

```python
# db/models.py - KnowledgeBase 创建时锁定三元组
embedding_provider: Mapped[str]      # "doubao"
embedding_model_id: Mapped[str]      # "doubao-embedding-vision-251215"
embedding_vector_dim: Mapped[int]    # 2048

# api/deps.py - 每次查询时严格校验
if provider.model_info.vector_dim != kb.embedding_vector_dim:
    raise EmbeddingModelMismatch(...)
    # → HTTP 409 Conflict
```

为什么是 409 而不是 400？
- 400 表示请求本身的格式/参数错误
- 409 Conflict 表示请求在当前**服务器状态**下无法完成——向量维度冲突是一种状态冲突，语义上更准确

**如果确实需要更换模型**

正确做法：
1. 创建新的 Knowledge Base（新 collection）
2. 重新上传/触发所有文档的重新索引（用新模型）
3. 删除旧 KB

这是有意识的强制操作，而不是静默改变。

**追问预判**

- "有没有在线迁移方案，不用重新入库？" — 理论上可以用模型蒸馏（将旧向量投影到新空间），但这需要专门的对齐训练，工程复杂度极高，实际项目中不建议。

---

### Q2-7：CrossEncoder Reranker 的原理，为什么比 Bi-encoder 更准确？二阶段检索的权衡

**答题思路**

从架构上解释 Bi-encoder vs CrossEncoder 的核心差异，再分析精度-延迟权衡，说明本系统的设计选择。

**核心答案**

**Bi-encoder（本系统 Dense Retrieval 使用的模式）**

```
Query  → Encoder → Q_vector
Doc    → Encoder → D_vector
Score  = cosine(Q_vector, D_vector)
```

Q 和 D 独立编码，向量可以**离线预计算**存储，查询时只需实时 encode Query。支持 ANN 索引，可以在百万文档中毫秒级检索。但独立编码意味着 Q 和 D 之间缺少**跨注意力**（cross-attention），无法建模 "query 的哪些词与 doc 的哪些词相互关联"。

**CrossEncoder**

```
[CLS] query [SEP] document [SEP]
         ↓
   Transformer (所有层都能跨 Q 和 D 做注意力)
         ↓
   [CLS] token → Linear → score
```

Query 和 Document 在同一个 Transformer 中联合编码，所有 self-attention 层都能建模 Q 和 D 之间的精细交互。代价：**无法预计算 Doc 向量**，每次查询都要把 (query, doc) 重新 encode，复杂度是 O(N)。

**本系统的二阶段设计**

```
Stage 1（召回）：Bi-encoder + BM25 → top-20 候选（快，~10ms）
Stage 2（精排）：CrossEncoder rerank → top-8 精排（慢，~200ms，仅对 20 个候选）
```

用 Stage 1 把候选从全库（可能 10 万条）缩小到 20 条，CrossEncoder 只处理 20 个，时间复杂度从 O(N) 降为 O(prefetch_limit)。

**本系统实现**

```python
# retriever.py - CrossEncoder 作为可选模块
class CrossEncoderReranker:
    def __init__(self, model_name="ms-marco-MiniLM-L-6-v2"):
        # FlashRank 的 ONNX 实现，无需 GPU，CPU 上 ms 级
        from flashrank import Ranker
        self._ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank")
```

选择 `ms-marco-MiniLM-L-6-v2` 的理由：在 MS MARCO（信息检索 benchmark）上训练，6 层 MiniLM 比全量 BERT 小 4 倍，CPU 推理延迟约 50-200ms/20候选，精度相当。

**追问预判**

- "reranker 一定比 bi-encoder 准确吗？" — 不一定。如果 Stage 1 召回已经足够好（reranker 需要的答案在 top-20 里），reranker 确实能提升精度。但如果答案在 top-20 之外（召回不足），reranker 无法挽救。所以先要保证召回，再做精排。
- "有没有比 MiniLM 更强的 reranker？" — BGE-reranker-large、bge-reranker-v2-m3 等，精度更高但速度慢 3-5 倍。

