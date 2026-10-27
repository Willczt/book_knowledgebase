# RAG 效果评估手册（可复现）

> 目标：让「效果好不好」变成**可测量的数**，而不是感觉。
> 本文所有数字都来自本仓库 2026-10-27 的实测（真实知识库：26 本书 / 35941 chunks / bge-m3 1024 维）。

---

## 0. 一句话结论

**100 条评测集上：无精排 Hit@1 = 0.608 / Recall@10 = 0.809 / MRR@10 = 0.709；开启精排后 Hit@1 = 0.763 / Recall@10 = 0.887（+15.5pt / +7.8pt）。生成侧 faithfulness 3.75/5、拒答率 100%、误拒率 4.8%、TTFT P50 约 4.8s。**

四个能讲清楚的结论：
1. **精排是这套系统里性价比最高的一环**：+15.5pt Hit@1，代价是每条查询多约 110ms。而且它主要修的是「排序」不是「召回」——多跳题的 R@30 几乎没变（0.691→0.691），但 MRR 从 0.654 → 0.800。
2. **有两个完全不同的「阈值」，不能混用**：`score_threshold` 作用在 RRF 融合分上（名次信号，量级 0.3~1.0），`min_rerank_score` 作用在 rerank 分上（0~1 相关度）。前者 0.9 时 Recall@10 从 0.809 掉到 0.30。
3. **「低于阈值就不可信」不能靠单分数**：正负样本分数分布重叠严重（负样本 P99=0.949 高于正样本 P50=0.833），硬阈值会大量误杀。标定结果显示它适合「降噪省 token」，不适合「判不可信」。
4. **精排可能静默失效**：服务不可达时，指标与「关闭精排」完全一致，用户毫无感知——所以每次做对照实验前必须先确认 `rerank.status=applied`。

---

## 1. 评测做在哪一层（四层指标，各有各的用途）

| 层 | 指标 | 实测值 | 采集成本 | 用途 |
|---|---|---|---|---|
| **检索** | Hit@1 / Recall@K / MRR@10 / NDCG@10 | 见下表 | 极低（不调 LLM，30 条 ≈ 5 秒） | 调分块、embedding、阈值、精排 |
| **生成** | faithfulness / correctness / relevance | 3.94 / 4.06 / 4.52 | 中（LLM judge，33 条 ≈ 4 分钟） | 调 Prompt、模型、上下文组织 |
| **生成** | 拒答率 / 误拒率 | 100% / 0% | 低（规则判定） | 防幻觉的核心指标 |
| **生成** | TTFT / 总耗时 / token | 4.79s / 6.60s（P50） | 低（埋点） | 体验优化 |
| **系统** | 摄入成功率、失败重试、P95 延迟、可达性 | 26/26 indexed | 低（DB + trace） | 稳定性 |
| **业务** | 坏 case 归因分布、用户点踩率 | — | 需埋点 | 决定优化优先级 |

> **面试要点**：RAG 的失败大多在检索层，但检索指标最便宜。所以正确顺序是「先把检索指标跑通，再看生成」。

### 1.1 检索基线（100 条，`use_reranker=false`，`top_k=30`）

| 切片 | n | Hit@1 | R@3 | R@5 | R@8 | R@10 | R@30 | MRR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **全部** | 97 | 0.608 | 0.742 | 0.784 | 0.809 | 0.809 | 0.918 | 0.709 | 0.708 |
| exact_term | 13 | 0.538 | 0.769 | 0.846 | 0.846 | 0.846 | 1.000 | 0.673 | 0.717 |
| image | 21 | 0.714 | 0.809 | 0.905 | 0.905 | 0.905 | 1.000 | 0.783 | 0.813 |
| multi_hop | 21 | 0.524 | 0.452 | 0.524 | 0.548 | 0.595 | 0.691 | 0.654 | 0.520 |
| paraphrase | 27 | 0.667 | 0.815 | 0.852 | 0.852 | 0.852 | 0.889 | 0.736 | 0.765 |
| single_hop | 15 | 0.533 | 0.733 | 0.800 | 0.867 | 0.867 | 0.933 | 0.661 | 0.712 |

**从这张表能直接读出的决策**：
- **R@8 = R@10 = 0.809**，说明 `top_k=8` 之后基本没有新增命中；把 top_k 提到 30 只多 10.9 个点召回，但上下文 token 成本涨 3.75 倍。→ **默认 top_k=8 是合理的**。
- **multi_hop 是唯一明显掉队的题型**（Hit@1 0.524、NDCG 0.520）：它有 2 个 gold，单次检索很难同时命中两本书——这是 Agentic 改造唯一有明确数据支撑的动机。
- **exact_term 的 R@30=1.0、Hit@1 只有 0.538**：专有名词召回没问题，是排序问题 → 精排该解决它（下节验证：0.538 → 0.923）。

### 1.2 精排 on/off 对照（同一 97 条，`top_k=30`）

| 指标 | 无精排 | 精排 on | 差值 |
|---|---|---|---|
| Hit@1 | 0.608 | **0.763** | +15.5pt |
| R@10 | 0.809 | **0.887** | +7.8pt |
| R@30 | 0.918 | 0.918 | 0（精排不改变候选池） |
| MRR@10 | 0.709 | **0.847** | +13.8pt |
| NDCG@10 | 0.708 | **0.824** | +11.6pt |

分切片 Hit@1：

| 切片 | 无精排 | 精排 on | 差值 |
|---|---|---|---|
| exact_term | 0.538 | 0.923 | **+38.5pt** |
| image | 0.714 | 0.905 | +19.0pt |
| single_hop | 0.533 | 0.733 | +20.0pt |
| multi_hop | 0.524 | 0.667 | +14.3pt |
| paraphrase | 0.667 | 0.667 | 0（噪声） |

**结论**：精排不是「可有可无的加分项」，是主链路。R@30 完全没变而 MRR 涨 13.8pt，说明它只做重排、不扩召回——这也意味着**精排无法拯救「gold 不在候选池」的问题**（多跳题 R@30 只有 0.691，上限就在那）。

---

## 2. 评测集怎么建（已落地 100 条）

### 2.1 现有构成

| 类型 | 数量 | 说明 |
|---|---|---|
| `paraphrase` | 27 | 口语化改写，原文用词几乎不出现 |
| `multi_hop` | 21 | 跨两本书的比较/综合题，gold 2 个片段 |
| `image` | 21 | 图片题，gold 为图片 caption 片段（已排除封面/版权页） |
| `single_hop` | 15 | 单点事实 |
| `exact_term` | 13 | 依赖专有名词/数字 |
| `no_answer` | 3 | 库里没有答案，考拒答（不参与检索指标） |

字段：`question / type / gold_chunk_ids / answer_points / source_doc / expected_refusal`。

> 总 100 条来自 26 本书、35941 个 chunk。题型覆盖是有意设计的：如果只测单跳事实题，
> 会高估系统能力（实测单跳 R@30=0.933，而多跳只有 0.691）。

### 2.2 生成方式：反向造题（半自动，三种模式）

```bash
# 单跳（含 exact_term / paraphrase 变体）
.venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --kb-id xxx \
  --mode single --limit 25 --per-doc 2 --append

# 跨书多跳（gold 自动带上两个 chunk）
.venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --kb-id xxx \
  --mode multi_hop --limit 21 --append

# 图片题（读取 image_caption 片段，自动排除封面/版权页/宣传页）
.venv/bin/python scripts/gen_eval_set.py --collection kb_xxx --kb-id xxx \
  --mode image --limit 21 --per-doc 1 --append
```

流程：Qdrant 按文档分层抽样 → LLM 读片段写「用户会怎么问」→ 自动带上 `gold_chunk_id` → 输出 `eval/dataset.jsonl`。

**提示词里做了三件事**（很重要，否则指标虚高）：
1. 不许照抄原文超过 8 个连续字 → 否则 BM25 白送命中；
2. 问题必须能由该片段**完全**回答 → 保证 gold 唯一；
3. `temperature=0.7` → 措辞更接近真实用户。

**已知偏差（面试要主动说）**：
- 问题来自片段，天然"有解"，测不出「用户问的东西书里没有」的情况（所以补了 3 条 `no_answer`）；
- 与生成模型同源（DeepSeek），措辞分布可能偏同一风格；
- 30 条样本的置信区间很宽：Recall@10 的 95% 置信区间约 ±0.14，所以**差异小于 8 个百分点不要当真**。

### 2.3 扩到 100+ 的路径

| 来源 | 方法 | 成本 |
|---|---|---|
| 真实提问 | 前端加"点踩 + 一句话反馈"，每周抽样 | 需埋点 |
| 人工补题 | 按书目录挑章节，人工写 20 条/小时 | 低 |
| 多跳题 | 取两个不同文档的片段，让 LLM 出对比题（gold = 2 个 chunk） | 中 |
| 图片题 | 用 `image_caption` 片段造题，测多模态召回 | 中 |

---

## 3. 工具链（两条命令）

```bash
# ① 检索评测 / 参数扫描（不花 LLM 额度）
.venv/bin/python scripts/eval_rag.py \
  --dataset eval/dataset.jsonl \
  --kb-id 0569a988-80a5-49d9-8aaa-5486a1aeda8f \
  --base-url http://127.0.0.1:8001 \
  --stage retrieval --top-k 30 --no-rerank --label baseline

# 扫描
... --sweep "threshold=0,0.3,0.5,0.6"
... --sweep "rerank=on,off"
... --sweep "top_k=5,8;threshold=0,0.3"

# ② 生成质量（含拒答 + LLM judge）
.venv/bin/python scripts/eval_rag.py ... --stage generation \
  --gen-top-k 8 --judge --llm-provider deepseek --llm-model deepseek-flash
```

产物：`eval/reports/<label>_<时间戳>.md` + 同名 `.json`（含逐条 rank、回答、判分，可离线分析）。
指标口径有自检：`--self-test`。

**与后端已有能力的衔接**（面试可讲"不是另起一套"）：
- `trace.ranking_metrics` 提供同样的 Recall/MRR/NDCG 计算；
- `POST /debug/traces/{id}/evaluate` 可以对单条线上请求补标注；
- `stage_metrics` 提供各阶段 P50/P95；
- 评测脚本相当于把这些能力**批量化、可对比**。

---

## 4. 能调哪些参数（按"调完多久能看到效果"分三类）

### A 类：改配置重启即可验证（分钟级）

| 参数 | 当前值 | 实测影响 | 建议扫描范围 |
|---|---|---|---|
| `DEFAULT_TOP_K` | 8 | R@8 已饱和（0.80），加到 30 只 +10 个点，上下文成本 ×3.75 | 5 / 8 / 12 |
| `score_threshold`（召回层） | 0 | 作用于 **RRF 融合分**（名次信号）；0.9 时 R@10 从 0.809 → 0.30 | 0 / 0.3 / 0.5 |
| `min_rerank_score`（精排层，新增） | 0 | 作用于 **rerank 分**（0~1 相关度）；0.05 → 延迟 -17%、误拒率翻倍，质量在噪声内（见 5.5） | 0 / 0.02 / 0.05 |
| `RERANK_ENABLED` | true | 实测：关掉后 Hit@1 0.763→0.608（-15.5pt） | on / off |
| `RERANK_CANDIDATES` | 30 | 候选池越大精排越准但线性变慢；当前 30 条精排 P50 ~113ms | 15 / 30 / 50 |
| `RERANK_MODEL` | bge-reranker-v2-m3 | 中文效果好；换模型需重跑对照 | — |
| `QUERY_REWRITE` | llm | 只影响多轮追问；首轮无收益还多耗 ~28ms + token | llm / concat / off |
| `LLM_MODEL` | deepseek-flash | 影响 correctness / faithfulness / TTFT | 对比 deepseek-chat |
| `LLM_TEMPERATURE` | 0.7 | RAG 场景 0.1~0.3 更稳（降低自由发挥） | 0.1 / 0.3 / 0.7 |
| `LLM_MAX_TOKENS` | 8192 | 太小会"只思考不出正文"（推理模型） | 4096 / 8192 |
| `CAPTION_PROVIDER` | deepseek | 影响图片描述质量 → 影响图片检索 | deepseek / ollama |

### B 类：必须重建知识库（小时级，成本高，一次只做一个）

| 参数 | 当前值 | 影响面 | 实验设计 |
|---|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | 召回粒度、上下文完整性、成本 | 建两个 KB：512 vs 800，同一批文档 + 同一评测集 |
| Embedding 模型 | ollama/bge-m3 (1024) | **召回质量的最大变量** | 新建 KB 换 nomic-embed-text(768) 或 doubao 对比 |
| 多模态开关 | off | 图文混排书籍的图片召回 | 开/关各建一个库，用图片题评测 |
| `MAX_IMAGE_DIMENSION` | 1024 | caption 质量 vs 成本 | 768 / 1024 |
| 解析器策略 | 默认 | TXT 章节识别、EPUB 去重 | 抽 3 本书人工对比 chunk 边界 |

> **为什么不能在线调**：KB 创建时锁定 `(provider, model_id, vector_dim)`，不同模型的向量空间不可混用（409 保护）。这是设计上的正确取舍——宁可让实验麻烦，也不让线上静默错。

### C 类：要改代码（天级）

- 检索权重：当前 RRF 等权，可改加权融合（dense 0.7 / sparse 0.3）；
- 上下文组装：当前按 `top_k` 条拼接，无 token 预算/去重/压缩（`min_rerank_score` 已能砍掉低分尾巴）；
- 引用约束：Prompt 里没有强制 `[1][2]` 标注，导致引用覆盖率指标失效；
- 多路召回：多跳题 R@30 只有 0.691，需要查询分解/多查询扩展才能突破。

> 已完成的 C 类改动：**阈值过滤已支持在精排后生效**（`min_rerank_score`），
> 标定方法与实测结果见 5.5。

---

## 5. 校验方法：怎么确认「调了之后真的变好」

### 5.1 标准流程（单变量）

```
1. 记录基线：同一数据集、同一后端配置 → 存 baseline 报告
2. 只改一个参数 → 重启 → 跑同一数据集 → 存新报告
3. 对比：看 Overall 和 by_type 两级；差异 < 8 个百分点的当噪声
4. 看代价：延迟、token、成本是否可接受
5. 记决策：改了什么、为什么、数据支撑
```

### 5.2 召回阈值扫描（100 条，精排开启，只改 `score_threshold`）

| threshold | Hit@1 | R@10 | MRR@10 | 实际含义 |
|---|---|---|---|---|
| 0 | **0.763** | **0.887** | **0.847** | 不过滤（推荐） |
| 0.3 | 0.732 | 0.763 | 0.777 | 砍掉单路 rank≥3 的候选 |
| 0.5 | 0.713 | 0.692 | 0.739 | 只保留双路靠前的 |
| 0.6 | 0.598 | 0.571 | 0.603 | 只剩双路都进前 2 的 |
| 0.9 | 0.370 | 0.342 | 0.370 | 几乎只剩两路榜首 |

**结论**：
- 融合分是 RRF 的 `1/(rank+1)` 组合，**量级 0.3~1.0，与语义相似度无关**；所以"设阈值 0.45 提升质量"是错的直觉，实际是"只要前 2 名"，R@10 直接掉 19.5 个点。
- 想要按相关度过滤，必须用精排分（0~1），已实现为独立参数 `min_rerank_score`，见 5.5。
- 这也是为什么两个阈值要分开命名：混用一个参数会让"调阈值"变成"改召回条数"。

### 5.3 本轮实测：精排失效检测

同一数据集，`use_reranker=true`（服务不可达）：

```
baseline : Hit@1 0.600 | R@10 0.800 | MRR 0.675 | NDCG 0.706 | P50 106.2ms
rerank   : Hit@1 0.600 | R@10 0.800 | MRR 0.675 | NDCG 0.706 | P50 105.4ms
```

**指标一模一样 = 精排没生效**。翻 trace 得到证据：

```
rerank: cooldown | enabled: false
reason: 上次重排请求失败，冷却期内直接沿用融合分排序
error : rerank 服务处于失败冷却期
```

→ 这就是「A/B 实验里对照组和实验组完全一致」的典型陷阱：不是参数没用，是参数没生效。**做对照实验前必须先确认实验组真的生效**（看 trace 的 `status=applied`）。

### 5.4 生成侧实测（24 条混合题型，deepseek-flash，精排开启）

| 指标 | 值 | 解读 |
|---|---|---|
| faithfulness | 3.75 / 5 | 有若干条 ≤3，典型是「补了资料里没有的细节」 |
| correctness | 4.04 / 5 | 答案要点覆盖尚可 |
| relevance | 4.33 / 5 | 基本不跑题 |
| 拒答率（3 条无答案题） | 100% | 全部正确说"资料里没有" |
| 误拒率（21 条有答案题） | 4.8%（1 条） | 有一次过度保守 |
| 引用覆盖率 | 0.031 | **指标失效**（见下） |
| TTFT P50 / P95 | 4.79s / 15.4s | 交互体验的瓶颈 |
| 总耗时 P50 / P95 | 6.60s / 22.5s | 主要花在生成 |

**两个值得深挖的失败样本**：
- `q002`：答案说《蓝皮书》，但资料里只有"布劳威尔讲座" → judge 判 faithfulness=3。属于**模型用世界知识补全**。
- `q013`：faithfulness=4 但 correctness=1 → 回答忠实于资料，但资料本身不是答案 → **检索召回到了相关但非答案的片段**，这是检索层问题，不是生成层。

→ 这就是为什么必须**分层归因**：否则会把检索问题误判成"模型不行"，然后去换更贵的模型。

### 5.5 可信分阈值怎么定（标定方法 + 实测）

**问题**：「召回的可信分低于阈值就认为不可信」——这个阈值从哪来？

**先分清两个分数**：

| 分数 | 谁产生 | 取值范围 | 能否当"可信度" |
|---|---|---|---|
| 融合分 `score_threshold` | Qdrant RRF | 0.3~1.0（`1/(rank+1)` 组合） | **不能**，它只反映名次，榜首几乎总是 1.0 |
| 精排分 `min_rerank_score` | CrossEncoder | 0~1（真实相关度） | 可以，但需要标定 |

**标定方法**（`scripts/calibrate_threshold.py`）：

1. 对评测集每道题取 top-30 候选（不过滤），记录每个候选的 rerank 分；
2. 按 gold 标注切成**正样本**（gold 片段）和**负样本**（其余片段）；
3. 看两组分布，扫阈值，算两个量：
   - **gold 保留率**：阈值下有多少正样本被留下（漏答风险）
   - **噪音剔除率**：有多少负样本被砍掉（省 token 收益）
4. 按业务代价选工作点。**不报「精确率」**——评测集只标了答案，没标「一定无关」，未标注 ≠ 无关。

**实测（100 条 / 正样本 101 / 负样本 2809）**：

| 分位 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|---|---|
| 正样本 | 0.073 | 0.122 | 0.384 | 0.833 | 0.997 | — | — | — |
| 负样本 | — | — | — | 0.077 | 0.174 | 0.420 | 0.657 | 0.949 |

**关键事实：负样本 P99（0.949）高于正样本 P50（0.833）** —— 两组严重重叠。

| 阈值 | gold 保留率 | 噪音剔除率 | 平均保留条数 |
|---|---|---|---|
| 0（不过滤） | 100% | 0% | 30.0 |
| **0.02（保守）** | **98.0%** | **28.1%** | 21.9 |
| **0.05（激进）** | **95.0%** | **36.3%** | 19.4 |
| 0.3 | 85.3% | 82.0% | 6.1 |
| 0.5 | 74.7% | 88.9% | 4.0 |

**生成侧验证（24 条，只改 `min_rerank_score`）**：

| 指标 | 0 | 0.05 | 结论 |
|---|---|---|---|
| faithfulness | 3.75 | 3.83 | +0.08，噪声 |
| correctness | 4.04 | 4.08 | +0.04，噪声 |
| 误拒率 | 4.8% | 9.5% | **翻倍**（1→2 条） |
| TTFT P50 | 4.79s | 3.92s | **-18%** |
| 总耗时 P50 | 6.60s | 5.49s | **-17%** |

**结论（面试可以直接说）**：
1. 阈值 **0.02~0.05** 的真实收益是**降延迟、省上下文**（-17% 时延、上下文条数 30→19），不是提质量；
2. 想用它做「不可信就拒答」，会大量误杀（负样本 P99 比正样本中位数还高）——拒答需要额外信号：多查询一致性、跨片段互证、模型自评、或证据不足时显式说"资料里没有"；
3. 阈值必须**随 embedding/rerank 模型换而重标**，不同模型的分数分布不可迁移；
4. 生产上还要看业务代价：客服场景漏答代价高（选 0.02），探索型问答可以用 0.05。

### 5.6 常见误判（面试高频追问）

| 陷阱 | 表现 | 防范 |
|---|---|---|
| 参数没生效 | 两组指标完全一致 | 先验证 `status=applied` / 配置快照写入报告 |
| 数据集泄漏 | 问题含原文短语，R@10 虚高 | 提示词禁止照抄；人工抽查 10 条 |
| 样本太小 | 改动 3 个点就"提升" | 差异 < 8 个点视为噪声；扩到 100+ 再判断 |
| 用错指标 | 引用覆盖率 0.011 当"引用能力差" | 指标要先验证敏感性，再用于结论 |
| judge 漂移 | 换了 judge 模型，分数全变 | 固定 judge 版本；用 30 条人工标注校准（一致率 ≥80%） |
| 只测平均 | 平均值没变，长尾爆炸 | 同时看 P95 与 by_type 切片 |

---

## 6. 上线后的生产观测（离线测不出真实分布）

### 6.1 必埋的 8 个点

| 指标 | 采集位置 | 告警阈值建议 |
|---|---|---|
| 检索 top1 分 / 命中条数 | `search` 响应 | top1 < 0.5 的占比 > 30% |
| 精排状态 | trace `rerank.status` | `failed/cooldown` 占比 > 5% 持续 5 分钟 |
| 无答案触发率 | 回答文本 + judge | 突增 2 倍（可能索引损坏） |
| 空回答率 | `LLMEmptyResponse` | > 1% |
| TTFT P95 | chat 流首个 token | > 8s 持续 10 分钟 |
| 摄入失败率 | `documents.status=failed` | > 5% |
| 队列积压 | `pending` 数量 | 持续增长 30 分钟 |
| 用户负反馈率 | 点踩 / 总提问 | > 5% |

### 6.2 灰度发布（企业必答）

1. 新配置先在**影子 KB**（小流量文档集）跑离线评测；
2. 用 `tenant_id`/KB 维度做流量分组，双跑对比线上指标；
3. 观察 24 小时（覆盖不同查询分布），再全量；
4. 保留一键回滚（参数在 `.env`/DB，改回即生效，无需重建索引）。

### 6.3 与 SLO 挂钩

| SLO | 目标 | 依据 |
|---|---|---|
| 检索可用性 | 99.9% | `/readyz` |
| 检索 P95 | < 500ms | 实测 159ms，留 3 倍余量 |
| 问答 TTFT P50 | < 5s | 实测 4.79s |
| 问答 TTFT P95 | < 10s | 实测 15.4s（长回答尾部，需优化） |
| 摄入成功率 | > 95% | 当前 26/26 |
| 拒答正确率 | > 90% | 实测 100%（3 条样本太小，需扩） |

---

## 7. 面试怎么讲这段（三层递进）

**第一层（有没有做过）**
> 我搭了一套评测：33 条自建数据集（含 3 条无答案），检索层用 Recall@K/MRR/NDCG，生成层用 LLM judge 打 faithfulness/correctness，还单独测拒答率。检索评测 30 条只要 5 秒，所以可以高频做参数对照。

**第二层（会不会分析）**
> 我拿到的基线是 Hit@1 0.60、R@10 0.80。看切片发现 single_hop 的 Hit@1 只有 0.50 但 R@30 有 0.90——说明不是召回不到，是排序不准，该修精排而不是改分块。另外阈值扫描发现 `score_threshold` 用的是 RRF 融合分，0.9 时 R@10 直接从 0.80 掉到 0.30，所以那些"调阈值提精度"的经验在这套系统上不成立。

**第三层（能不能落到生产）**
> 离线只能证明相对优劣，真实分布要靠在线观测：我在 trace 里记了精排状态、每阶段耗时、引用情况，可以按周看坏 case 归因分布。上线走影子 KB + 小流量双跑，指标掉超过阈值就回滚。

**高频追问预案**：

- *「你的评测集是不是自己造的，有偏？」* → 是反向生成，坦诚说明三个偏差（有解偏置、同源模型、样本 30 条），并说扩量路径（真实点踩抽样 + 人工补题）。
- *「LLM judge 可信吗？」* → 需要校准：30 条人工标注对齐，一致率 <80% 就不用；只判"是否被资料支持"，不判文采。
- *「怎么证明优化有效？」* → 单变量 + 同数据集 + 差异阈值 8 个点 + 检查实验组真的生效（我踩过精排静默失效、两组指标一模一样的坑）。
- *「为什么不做端到端 A/B？」* → 现在没有流量和用户反馈埋点，是下一步；当前用离线评测做筛选，线上只做灰度验证。

---

## 8. 下一步（两周计划，按投入产出排序）

**已完成（本轮）**：
- ✅ 精排 on/off 对照：Hit@1 +15.5pt，证明精排值得保留；
- ✅ 阈值过滤支持精排后生效（`min_rerank_score`），并做了标定（5.5）；
- ✅ 评测集扩到 100 条（21 多跳 + 21 图片）；
- ✅ 修掉评测暴露的生产 bug：SSE 客户端断开时 DB 会话清理失败会拖垮连接池（见附录）。

| 优先级 | 事项 | 预期产出 |
|---|---|---|
| P0 | 引用改为结构化 `[1][2]` + 后端校验 | 替代失效的字符串覆盖率指标（当前 0.03） |
| P0 | 多跳题专项：查询分解 / 多路召回 | R@30 从 0.691 往上（这是唯一有数据支撑的 Agentic 动机） |
| P1 | 前端点踩 + 归因埋点 | 拿到真实分布，形成线上评测闭环 |
| P1 | judge 人工校准（30 条） | judge 分数可对外引用 |
| P1 | 精排阈值接入对话默认配置（当前默认 0） | 把 -17% 延迟落地到线上 |
| P2 | CI 冒烟评测（10 条，Recall@10 掉 5 个点即失败） | 防止回归 |
| P2 | 上下文 token 预算（替代固定 top_k） | 长文档场景降 TTFT |

---

## 8.1 附录：评测顺带修掉的一个生产 bug

跑批量评测时后端偶发 500，追下去是：

```
sqlalchemy.exc.IllegalStateChangeError: Method 'close()' can't be called here;
method '_connection_for_bind()' is already in progress
```

**触发条件**：SSE 流式响应 + 客户端读完 `done` 后立即断开（评测脚本、浏览器关闭标签页都会）。
请求被取消时，`Depends(get_session)` 的清理正好撞上 SQLAlchemy 的 greenlet 切换，
`close()` 抛异常 → 连接不归还连接池 → 连续几次后池被占满 → **整个服务假死**。

**修复**（`db/engine.py` + `api/deps.py`）：
1. `get_session` 不再用 `async with`，改为显式 `close()`，异常时 `invalidate()` 丢弃连接；
2. `get_db` 改为显式 `anext/aclose`，保证依赖清理一定走到。

**验证**：连续 10 次 SSE 问答，`IllegalStateChangeError` 计数为 0，`/knowledge-bases` 持续 200。

> 面试价值：这是"评测不是为了出报告，而是能逼出生产缺陷"的实例——
> 单次要人肉点几下页面根本复现不了，跑 100 条批处理立刻就炸。

---

## 9. 复现清单（换机器照做）

```bash
# 前置：Ollama 跑起来（KB 用 bge-m3）+ rerank 服务可达 + 后端可访问
curl -s localhost:11434/api/tags | grep bge-m3
curl -s -o /dev/null -w "%{http_code}\n" http://192.168.18.183:8100/v1/models   # 期望 200

# 0) 指标自检
.venv/bin/python scripts/eval_rag.py --self-test

# 1) 造题（三种模式，可重复 append）
.venv/bin/python scripts/gen_eval_set.py --collection <kb_collection> --kb-id <kb_id> --mode single --limit 25
.venv/bin/python scripts/gen_eval_set.py --collection <kb_collection> --kb-id <kb_id> --mode multi_hop --limit 21 --append
.venv/bin/python scripts/gen_eval_set.py --collection <kb_collection> --kb-id <kb_id> --mode image --limit 21 --per-doc 1 --append

# 2) 基线（无精排）
.venv/bin/python scripts/eval_rag.py --dataset eval/dataset.jsonl --kb-id <kb_id> \
  --base-url http://127.0.0.1:8001 --stage retrieval --top-k 30 --no-rerank --label baseline

# 3) 精排 on/off（先确认实验组真的生效，看 trace 的 rerank.status=applied）
... --rerank --label rerank-on

# 4) 参数对照（每次只改一项）
... --sweep "threshold=0,0.3,0.5"          # 召回层（RRF 融合分）
... --rerank --sweep "min_rerank=0,0.02,0.05"   # 精排层（rerank 分）

# 5) 阈值标定（找可信分下限）
.venv/bin/python scripts/calibrate_threshold.py --dataset eval/dataset.jsonl \
  --kb-id <kb_id> --base-url http://127.0.0.1:8001

# 6) 生成质量（含拒答项）
... --stage generation --gen-top-k 8 --rerank --judge --label gen
```

> 环境注意：本机 8000 端口被另一个 vLLM（Qwen3-8B）占用，后端跑在 **8001**。
> `.env` 的 `RERANK_BASE_URL` 指向 `192.168.18.183:8100`——它**曾经不可达**，
> 那时所有「有精排」实验实际都是降级态（两组指标完全一致），这正是 5.3 节的证据来源。
> 现在服务已恢复，跑之前先 `curl` 确认 200，否则会白跑一轮。

---

## 10. 人工复测清单（不想写代码时按这个走）

### 10.1 五分钟冒烟

```bash
# ① 三个服务都在
curl -s localhost:11434/api/tags | grep -q bge-m3 && echo "ollama ok"
curl -s -o /dev/null -w "rerank %{http_code}\n" http://192.168.18.183:8100/v1/models
curl -s -o /dev/null -w "backend %{http_code}\n" http://127.0.0.1:8001/api/v1/health

# ② 指标口径没被改坏
.venv/bin/python scripts/eval_rag.py --self-test

# ③ gold 数据没过期（KB 更新后必做）
.venv/bin/python scripts/eval_ci.py --kb-id <kb> --collection <collection> --check-only
```

### 10.2 单条人工对照（验证某个参数是否真的生效）

```bash
KB=<kb_id>
# A：关精排
curl -s -X POST "http://127.0.0.1:8001/api/v1/knowledge-bases/$KB/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"曾国藩为什么强调先自固","top_k":5,"use_reranker":false}' | python -m json.tool

# B：开精排 + 精排阈值 0.05
curl -s -X POST "http://127.0.0.1:8001/api/v1/knowledge-bases/$KB/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"曾国藩为什么强调先自固","top_k":5,"use_reranker":true,"min_rerank_score":0.05}' | python -m json.tool
```

看三件事：
1. B 的 `score` 是 rerank 分（0.9x 量级），A 的是融合分；
2. B 返回条数可能少于 A（阈值生效）；
3. 顺序应该变化——如果完全一样，先怀疑精排没生效。

然后开调试，看 trace：

```bash
curl -s -X PUT http://127.0.0.1:8001/api/v1/debug/status \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"layers":["query","retrieval","rerank","generation","performance"]}'
# 前端 /debug 页面 → 最新 trace → 「重排 Rerank」面板
# 必须看到 status=已生效；若是「冷却中/调用失败」，说明服务侧有问题
```

### 10.3 对话层人工验证（3 类问题各一条）

| 类型 | 例子 | 期望 |
|---|---|---|
| 有答案 | "曾国藩为什么强调先自固？" | 回答带来源，内容可核 |
| 无答案 | "我们公司 2025 年 Q3 营收是多少？" | 明确说资料里没有，不要编 |
| 多跳 | "《亮剑》和《遥远的救世主》对'强者逻辑'的看法有何不同？" | 引用两本书；若只引一本，属已知短板 |

判定标准（不要靠感觉）：
- 有答案题：回答是否覆盖 `answer_points`（数据集里那条）；
- 无答案题：是否拒答（出现"资料中没有/未提及"）；
- 任何题：引用片段能不能支撑结论（点开来源对照）。

---

## 11. 知识库更新后的自动化回归

### 11.1 先理解一个坑

Qdrant 的 point id 是 `uuid4`，**重新摄入任何一篇文档，它的 chunk id 就全变了**。
评测集里的 `gold_chunk_ids` 直接失效 → 指标暴跌，但系统其实没变差。
所以回归的第一步永远是「gold 体检」，不是跑分。

### 11.2 一键回归

```bash
# 体检（不跑分，秒级）
.venv/bin/python scripts/eval_ci.py \
  --kb-id <kb> --collection kb_xxx --check-only
# 退出码：0 全部存在 / 2 有失效

# 完整回归（体检 + 精排 on/off + 对比基线 + 出报告）
.venv/bin/python scripts/eval_ci.py \
  --kb-id <kb> --collection kb_xxx
# 退出码：0 通过 / 1 指标退化 / 2 gold 失效 / 3 服务不可用

# 确认这次就是新标准后，更新基线
.venv/bin/python scripts/eval_ci.py \
  --kb-id <kb> --collection kb_xxx --update-baseline
```

### 11.3 报告里应该有什么

`eval/reports/ci_<ts>.md` 固定包含 7 块：

1. **元信息**：时间、KB、collection、数据集指纹（SHA1 前 8 位）、样本数；
2. **数据体检**：gold 总数 / 缺失数 / 受影响题号 —— 这决定后面的数能不能信；
3. **检索指标（关精排）**：整体 + 分题型（Hit@1 / R@5 / R@10 / R@30 / MRR / NDCG）+ 延迟 P50/P95 + 失败数；
4. **检索指标（开精排）**：同上；
5. **与基线对比**：逐指标 delta + 是否超过门禁；
6. **结论一句话**：通过 / 退化 / 数据失效；
7. 原始 JSON（同名 `.json`）：逐条 ranked 列表，可离线复算。

> 为什么要固定这 7 块：报告是给"三个月后的自己"和"CI"看的。
> 缺了数据体检，你会对着一堆假指标调参；缺了分题型，你会漏掉多跳题退化。

### 11.4 出现 gold 失效怎么办

```bash
# 1) 看是哪些题失效
.venv/bin/python scripts/eval_ci.py ... --check-only    # 会打印 stale_case_ids

# 2) 找出这些题来自哪几本书
python -c "
import json
stale=set(['s001','s007'])   # 用上一步的输出
rows=[json.loads(l) for l in open('eval/dataset.jsonl',encoding='utf-8') if l.strip()]
print(sorted({r['source_doc'] for r in rows if r['id'] in stale}))
"

# 3) 只重造这几本书的题（把旧题从 dataset 里删掉再 append）
#    或者手工把 gold_chunk_ids 换成新 id（报告 JSON 的 per_case 里有上下文）
```

**经验**：如果分块参数（chunk_size/overlap）也改了，直接重造题更快，别手工修。

### 11.5 接进 CI

```yaml
# 伪代码：GitHub Actions / GitLab CI 都一样
- name: RAG regression
  run: |
    .venv/bin/python scripts/eval_ci.py --kb-id $KB --collection $COLLECTION
  # 退出码非 0 即失败；报告作为 artifact 上传
```

注意：CI 环境要有 Ollama（embedding）、Qdrant、rerank 服务，否则跑不了。
没有的话，把评测拆成「夜间任务」在固定机器上跑，而不是每次提交都跑。
