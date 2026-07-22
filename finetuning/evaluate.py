"""
微调效果评估脚本

对比微调前后的模型在 RAG 场景下的回答质量
"""
import argparse
import json
from pathlib import Path


def evaluate(
    lora_path: str = "outputs/lora",
    test_data: str = "data/test.jsonl",
    max_new_tokens: int = 512,
) -> None:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    test_samples = []
    with open(test_data, encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))

    print(f"评估 {len(test_samples)} 条样本...\n")
    results = []

    for i, sample in enumerate(test_samples[:20]):  # 最多评估 20 条
        instruction = sample["instruction"]
        context = sample.get("context", "")
        expected = sample["response"]

        prompt = (
            f"<|im_start|>system\n"
            "你是一个专业的知识库问答助手，请根据提供的参考资料准确回答问题。"
            f"<|im_end|>\n"
            f"<|im_start|>user\n参考资料：\n{context}\n\n问题：{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )
        generated = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "question": instruction,
            "expected": expected,
            "generated": generated,
        })
        print(f"[{i+1}] Q: {instruction[:60]}...")
        print(f"  期望: {expected[:100]}...")
        print(f"  生成: {generated[:100]}...")
        print()

    # 保存结果
    output_path = Path(lora_path) / "eval_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ 评估结果已保存到 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", default="outputs/lora")
    parser.add_argument("--test-data", default="data/test.jsonl")
    args = parser.parse_args()
    evaluate(args.lora_path, args.test_data)
