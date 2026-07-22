"""
将 LoRA 适配器导出为 GGUF 格式，供 Ollama 部署使用

用法：
  python export_to_gguf.py --lora-path outputs/lora --model unsloth/Llama-3.2-3B-Instruct

导出后通过以下命令部署到 Ollama：
  ollama create book-expert -f outputs/gguf/Modelfile
"""
import argparse
from pathlib import Path


def export(
    lora_path: str = "outputs/lora",
    base_model: str = "unsloth/Llama-3.2-3B-Instruct",
    output_dir: str = "outputs/gguf",
    quantization: str = "q4_k_m",
):
    from unsloth import FastLanguageModel

    print(f"加载 base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,  # Unsloth 从 LoRA 目录加载
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"导出 GGUF ({quantization})...")
    model.save_pretrained_gguf(
        str(output),
        tokenizer,
        quantization_method=quantization,
    )

    # 生成 Modelfile
    modelfile = output / "Modelfile"
    gguf_file = next(output.glob("*.gguf"), None)
    if gguf_file:
        modelfile.write_text(
            f"""FROM {gguf_file.name}

SYSTEM "你是一个专业的知识库问答助手，请根据提供的参考资料准确、简洁地回答问题。"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
""",
            encoding="utf-8",
        )
        print(f"\n✓ GGUF 文件: {gguf_file}")
        print(f"✓ Modelfile: {modelfile}")
        print(f"\n部署到 Ollama：")
        print(f"  ollama create book-expert -f {modelfile}")
        print(f"  ollama run book-expert")
    else:
        print("未找到 GGUF 文件，请检查导出是否成功")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", default="outputs/lora")
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--output", default="outputs/gguf")
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q5_k_m"],
        help="量化方法",
    )
    args = parser.parse_args()

    export(
        lora_path=args.lora_path,
        base_model=args.model,
        output_dir=args.output,
        quantization=args.quant,
    )
