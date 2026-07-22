"""
微调数据集准备脚本

从知识库 API 获取文档 chunk，使用强大的 LLM 生成问答对，
输出 SFT 训练数据集。

用法：
  python data/prepare_dataset.py --kb-id <id> --output data/train.jsonl
"""
import asyncio
import json
import os
import argparse
import httpx
from pathlib import Path

API_BASE = os.getenv("BOOKRAG_API_URL", "http://localhost:8000/api/v1")
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
DOUBAO_MODEL = os.getenv("DOUBAO_CHAT_MODEL", "doubao-seed-1-6-250715")

QA_GENERATION_PROMPT = """你是一个教育专家，请根据以下文段生成一个问答对。
要求：
1. 问题要有价值，能测试读者对内容的理解
2. 回答要准确、完整，直接基于原文
3. 回答不要泄露是从参考文段中提取的

文段：
{context}

请以 JSON 格式返回，例如：
{{"question": "...", "answer": "..."}}
只返回 JSON，不要其他内容。"""


async def generate_qa_pair(client: httpx.AsyncClient, context: str) -> dict | None:
    """调用 Doubao API 生成问答对"""
    try:
        resp = await client.post(
            f"{API_BASE.replace('/api/v1', '')}/api/v3/chat/completions".replace("localhost:8000", "ark.cn-beijing.volces.com"),
            headers={"Authorization": f"Bearer {ARK_API_KEY}"},
            json={
                "model": DOUBAO_MODEL,
                "messages": [
                    {"role": "user", "content": QA_GENERATION_PROMPT.format(context=context[:1000])}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # 解析 JSON
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"  QA 生成失败: {e}")
        return None


async def get_kb_chunks(kb_id: str) -> list[dict]:
    """从知识库获取所有文档的 chunk"""
    async with httpx.AsyncClient(base_url=API_BASE, timeout=60.0) as client:
        # 获取文档列表
        docs_resp = await client.get(f"/knowledge-bases/{kb_id}/documents?doc_status=indexed")
        docs = docs_resp.json()

        # 对每个文档做检索采样
        chunks = []
        for doc in docs:
            print(f"  采样文档: {doc['filename']} ({doc['chunk_count']} chunks)")
            # 使用空查询获取随机 chunk（作为训练数据种子）
            search_resp = await client.post(
                f"/knowledge-bases/{kb_id}/search",
                json={"query": "请介绍本文的主要内容", "top_k": 20},
            )
            results = search_resp.json().get("results", [])
            for r in results:
                if r.get("doc_id") == doc["id"] and r["chunk_type"] == "text":
                    chunks.append(r)

    return chunks


async def prepare_dataset(kb_id: str, output_path: str, max_samples: int = 500) -> None:
    print(f"从知识库 {kb_id} 准备训练数据...")
    chunks = await get_kb_chunks(kb_id)
    print(f"获取到 {len(chunks)} 个文本 chunk")

    # 去重
    seen_texts = set()
    unique_chunks = []
    for c in chunks:
        if c["text"] not in seen_texts and len(c["text"]) > 100:
            seen_texts.add(c["text"])
            unique_chunks.append(c)

    unique_chunks = unique_chunks[:max_samples]
    print(f"去重后 {len(unique_chunks)} 个 chunk，开始生成问答对...")

    samples = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, chunk in enumerate(unique_chunks):
            print(f"  [{i+1}/{len(unique_chunks)}] {chunk['source_doc'][:30]}...")
            qa = await generate_qa_pair(client, chunk["text"])
            if qa and qa.get("question") and qa.get("answer"):
                samples.append({
                    "instruction": qa["question"],
                    "context": chunk["text"],
                    "response": qa["answer"],
                    "source": chunk["source_doc"],
                })

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ 生成 {len(samples)} 条训练样本，保存到 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-id", required=True, help="知识库 ID")
    parser.add_argument("--output", default="data/train.jsonl", help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=500, help="最大样本数")
    args = parser.parse_args()

    asyncio.run(prepare_dataset(args.kb_id, args.output, args.max_samples))
