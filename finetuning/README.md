# Book Knowledgebase Finetuning Module

本模块提供针对垂直场景的本地 LLM 微调能力，采用 **QLoRA 4-bit + Unsloth** 方案，
显存占用比原始 LoRA 降低 ~75%，训练速度比原始 HuggingFace PEFT 快 2-3×。

## 工作流

```
知识库 RAG 数据 → 生成问答对 → 格式化 → LoRA 微调 → 导出 GGUF → Ollama 部署
```

## 目录结构

```
finetuning/
├── data/
│   ├── prepare_dataset.py   # 从知识库 API 生成 QA 对
│   ├── format_chat.py       # 格式化为 Chat 模板
│   └── train.jsonl          # (生成后存放)
├── configs/
│   └── lora_config.yaml     # LoRA 超参配置
├── outputs/
│   ├── lora/                # LoRA 适配器
│   └── gguf/                # 导出的 GGUF 文件
├── train.py                 # 训练入口
├── export_to_gguf.py        # 导出到 Ollama
├── evaluate.py              # 评估
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
cd finetuning
pip install -r requirements.txt
```

> **注意**: Unsloth 对 CUDA 版本有要求，推荐 CUDA 12.x + Python 3.10/3.11

### 2. 准备数据集

```bash
# 从知识库生成问答对（需要运行中的后端）
python data/prepare_dataset.py --kb-id <your-kb-id> --output data/train.jsonl

# 格式化为模型 Chat 模板
python data/format_chat.py --input data/train.jsonl --output data/formatted.jsonl --model-type llama3
```

### 3. 训练

```bash
# 默认：Llama-3.2-3B-Instruct，4-bit 量化
python train.py --model unsloth/Llama-3.2-3B-Instruct --data data/formatted.jsonl

# Qwen2.5-3B（中文能力更强）
python train.py --model unsloth/Qwen2.5-3B-Instruct --data data/formatted.jsonl --model-type qwen

# 完整参数
python train.py \
  --model unsloth/Llama-3.2-3B-Instruct \
  --data data/formatted.jsonl \
  --output outputs/lora \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4
```

### 4. 导出到 Ollama

```bash
python export_to_gguf.py --lora-path outputs/lora

# 然后部署
ollama create book-expert -f outputs/gguf/Modelfile
ollama run book-expert
```

### 5. 评估

```bash
python evaluate.py --lora-path outputs/lora --test-data data/test.jsonl
```

## 配置系统使用微调模型

微调并部署到 Ollama 后，在创建新会话时选择本地 Ollama 提供商，模型填写 `book-expert`。

## 硬件要求

| 模型规模 | 最低 VRAM | 推荐 VRAM |
|---------|----------|----------|
| 3B (4-bit) | 6 GB | 8 GB |
| 7B (4-bit) | 8 GB | 16 GB |
| 13B (4-bit) | 16 GB | 24 GB |

## 推荐基础模型

| 场景 | 推荐模型 | 说明 |
|------|---------|------|
| 中文问答 | `unsloth/Qwen2.5-7B-Instruct` | 中文理解最佳 |
| 速度优先 | `unsloth/Llama-3.2-3B-Instruct` | 轻量，推理快 |
| 质量优先 | `unsloth/Llama-3.1-8B-Instruct` | 综合能力强 |
