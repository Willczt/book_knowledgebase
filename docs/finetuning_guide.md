# 微调指南

本指南介绍如何使用 `finetuning/` 模块对本地模型进行垂直领域微调，使模型更擅长回答知识库中的内容。

---

## 为什么要微调？

通用 LLM 对专业领域（如技术类书籍、学术文献）的问答质量可能不足：
- 术语理解偏差
- 回答风格与文档领域不匹配
- 容易产生幻觉

通过在知识库数据上进行 **QLoRA 微调**（4-bit 量化 + LoRA），可以：
- 让模型熟悉领域术语
- 校准回答风格
- 显著降低幻觉率

---

## 完整微调流程

### 步骤 1：准备微调数据

```bash
cd finetuning

# 从知识库 API 生成 QA 对
python data/prepare_dataset.py \
  --kb-id <你的知识库ID> \
  --output data/train.jsonl \
  --max-samples 500

# 检查数据质量
head -5 data/train.jsonl | python -m json.tool
```

**数据样本格式**：
```json
{
  "instruction": "什么是 Transformer 的注意力机制？",
  "context": "Transformer 模型使用了自注意力机制...",
  "response": "Transformer 的注意力机制通过计算...",
  "source": "attention_is_all_you_need.pdf"
}
```

> 💡 建议至少准备 200 条高质量样本，500 条效果更佳。

### 步骤 2：格式化数据

根据目标模型类型格式化：

```bash
# Llama-3.x
python data/format_chat.py --model-type llama3

# Qwen-2.x（中文优先）
python data/format_chat.py --model-type qwen

# Mistral Instruct
python data/format_chat.py --model-type mistral
```

### 步骤 3：安装依赖

```bash
pip install -r requirements.txt
```

**GPU 要求**：需要支持 CUDA 的 NVIDIA GPU
- 最低 6GB VRAM（3B 模型 4-bit）
- 推荐 16GB VRAM（7B 模型 4-bit）

### 步骤 4：训练

```bash
# 中文领域（推荐）
python train.py \
  --model unsloth/Qwen2.5-3B-Instruct \
  --data data/formatted.jsonl \
  --output outputs/lora \
  --epochs 3

# 英文领域
python train.py \
  --model unsloth/Llama-3.2-3B-Instruct \
  --data data/formatted.jsonl \
  --output outputs/lora
```

训练完成后 LoRA 适配器保存在 `outputs/lora/`。

### 步骤 5：导出为 GGUF

```bash
python export_to_gguf.py \
  --lora-path outputs/lora \
  --output outputs/gguf \
  --quant q4_k_m   # 推荐量化：q4_k_m（平衡）/ q8_0（精度更高）
```

### 步骤 6：部署到 Ollama

```bash
# 创建 Ollama 模型
ollama create book-expert -f outputs/gguf/Modelfile

# 测试
ollama run book-expert "什么是注意力机制？"
```

### 步骤 7：配置系统使用微调模型

在前端创建新会话时：
- LLM 提供商：选择 `ollama`
- 模型 ID：填写 `book-expert`

或通过环境变量默认使用：
```env
OLLAMA_DEFAULT_MODEL=book-expert
```

---

## 性能参考

在 16GB VRAM 上训练 Qwen2.5-3B，500 条数据，3 epochs：

| 指标 | 参考值 |
|------|--------|
| 训练时间 | ~15 分钟 |
| 内存峰值 | ~14 GB |
| LoRA 适配器大小 | ~60 MB |
| GGUF q4_k_m 大小 | ~2 GB |

---

## 评估微调效果

```bash
# 准备测试集（20% 手动标注数据）
python evaluate.py \
  --lora-path outputs/lora \
  --test-data data/test.jsonl
```

评估结果保存在 `outputs/lora/eval_results.jsonl`，手动对比生成质量。

---

## 常见问题

### Q: OOM（显存不足）
```bash
# 减少 batch size，增加梯度累积步数
python train.py --batch-size 1 --gradient-accumulation-steps 16
```

### Q: 模型输出质量下降（过拟合）
- 减少 epochs（从 3 改为 1-2）
- 增加数据量（至少 500 条）
- 检查数据质量，过滤掉低质量 QA 对

### Q: Ollama 加载 GGUF 失败
- 确认 `ollama --version >= 0.3.0`
- 检查 GGUF 文件完整性（应大于 1GB）
