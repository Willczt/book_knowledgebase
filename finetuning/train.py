"""
使用 Unsloth + LoRA 对本地模型进行微调

用法：
  python train.py --model unsloth/Llama-3.2-3B-Instruct --data data/formatted.jsonl

默认配置适合 16GB VRAM；8GB VRAM 请减小 batch_size 到 1 并增大 gradient_accumulation_steps。
"""
import argparse
from pathlib import Path


def train(
    model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    data_path: str = "data/formatted.jsonl",
    output_dir: str = "outputs/lora",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print(f"加载模型: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    print("添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"加载数据集: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  训练样本数: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            save_steps=200,
            output_dir=output_dir,
            report_to="none",
            seed=42,
        ),
    )

    print("开始训练...")
    trainer_stats = trainer.train()
    print(f"训练完成！耗时: {trainer_stats.metrics['train_runtime']:.0f}s")

    # 保存 LoRA 适配器
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA 适配器已保存到 {output_dir}")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--data", default="data/formatted.jsonl")
    parser.add_argument("--output", default="outputs/lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true", help="不使用 4-bit 量化")
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        load_in_4bit=not args.no_4bit,
    )
