"""
将 SFT 数据集格式化为目标模型的 Chat 模板

支持：
- Llama-3 / Llama-3.2（ChatML 格式）
- Qwen-2.x（ChatML 格式）
- Mistral Instruct（[INST] 格式）

用法：
  python data/format_chat.py --input data/train.jsonl --output data/formatted.jsonl --model-type llama3
"""
import json
import argparse

ALPACA_TEMPLATE = """{system_prompt}

### Instruction:
{question}

### Context:
{context}

### Response:
{answer}"""

SYSTEM_PROMPT = (
    "你是一个专业的知识库问答助手，请根据提供的参考资料准确回答问题。"
    "只基于参考资料中的信息回答，不要凭空捏造。"
)


def format_alpaca(sample: dict) -> str:
    return ALPACA_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        question=sample["instruction"],
        context=sample.get("context", ""),
        answer=sample["response"],
    )


def format_chatml(sample: dict) -> str:
    context = sample.get("context", "")
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n参考资料：\n{context}\n\n问题：{sample['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['response']}<|im_end|>"
    )


def format_mistral(sample: dict) -> str:
    context = sample.get("context", "")
    return (
        f"[INST] {SYSTEM_PROMPT}\n参考资料：\n{context}\n\n问题：{sample['instruction']} [/INST]"
        f" {sample['response']}"
    )


FORMAT_FUNCS = {
    "alpaca": format_alpaca,
    "chatml": format_chatml,
    "llama3": format_chatml,
    "qwen": format_chatml,
    "mistral": format_mistral,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--output", default="data/formatted.jsonl")
    parser.add_argument(
        "--model-type",
        choices=list(FORMAT_FUNCS.keys()),
        default="llama3",
    )
    args = parser.parse_args()

    formatter = FORMAT_FUNCS[args.model_type]
    count = 0

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            text = formatter(sample)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

    print(f"✓ 格式化 {count} 条样本 → {args.output} (模板: {args.model_type})")


if __name__ == "__main__":
    main()
