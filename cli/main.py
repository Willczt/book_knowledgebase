"""
BookRAG CLI - 终端交互工具

使用示例：
  bookrag kb list
  bookrag kb create --name "ML Books" --embedding-provider doubao --model doubao-embedding-text-240715 --dim 2048
  bookrag ingest ./mybook.pdf --kb-id <id>
  bookrag search "什么是反向传播" --kb-id <id>
  bookrag chat --kb-id <id> --llm-provider doubao
  bookrag kb clear --kb-id <id>
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(name="bookrag", help="电子书 RAG 知识库 CLI 工具", rich_markup_mode="rich")
kb_app = typer.Typer(help="知识库管理")
app.add_typer(kb_app, name="kb")

console = Console()

DEFAULT_BASE_URL = os.getenv("BOOKRAG_API_URL", "http://localhost:8000/api/v1")


def get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=DEFAULT_BASE_URL, timeout=300.0)


# ========================
# KB 命令
# ========================

@kb_app.command("list")
def kb_list():
    """列出所有知识库"""
    async def _run():
        async with get_client() as client:
            resp = await client.get("/knowledge-bases")
            resp.raise_for_status()
            kbs = resp.json()

        table = Table(title="知识库列表", show_header=True)
        table.add_column("ID", style="dim", width=36)
        table.add_column("名称", style="bold")
        table.add_column("Embedding 模型")
        table.add_column("维度", justify="right")
        table.add_column("创建时间")

        for kb in kbs:
            table.add_row(
                kb["id"],
                kb["name"],
                f"{kb['embedding_provider']} / {kb['embedding_model_id']}",
                str(kb["embedding_vector_dim"]),
                kb["created_at"][:19],
            )
        console.print(table)

    asyncio.run(_run())


@kb_app.command("create")
def kb_create(
    name: str = typer.Option(..., "--name", help="知识库名称"),
    embedding_provider: str = typer.Option(..., "--embedding-provider", help="doubao|ollama|vllm"),
    model: str = typer.Option(..., "--model", help="Embedding 模型 ID"),
    dim: int = typer.Option(..., "--dim", help="向量维度"),
    multimodal: bool = typer.Option(False, "--multimodal", help="是否使用多模态 embedding"),
    description: Optional[str] = typer.Option(None, "--desc", help="描述"),
):
    """创建新知识库（锁定 embedding 模型）"""
    async def _run():
        async with get_client() as client:
            resp = await client.post("/knowledge-bases", json={
                "name": name,
                "description": description,
                "embedding_provider": embedding_provider,
                "embedding_model_id": model,
                "embedding_vector_dim": dim,
                "embedding_is_multimodal": multimodal,
            })
            resp.raise_for_status()
            kb = resp.json()

        console.print(Panel(
            f"[green]✓ 知识库创建成功[/green]\n"
            f"ID: [bold]{kb['id']}[/bold]\n"
            f"名称: {kb['name']}\n"
            f"Embedding: {kb['embedding_provider']} / {kb['embedding_model_id']} (dim={kb['embedding_vector_dim']})",
            title="知识库创建",
        ))

    asyncio.run(_run())


@kb_app.command("clear")
def kb_clear(
    kb_id: str = typer.Option(..., "--kb-id", help="知识库 ID"),
    confirm: bool = typer.Option(False, "--yes", help="跳过确认"),
):
    """清空知识库的所有向量（保留元数据）"""
    if not confirm:
        typer.confirm(f"确定要清空知识库 {kb_id} 的所有向量吗？此操作不可逆！", abort=True)

    async def _run():
        async with get_client() as client:
            resp = await client.post(f"/knowledge-bases/{kb_id}/clear")
            resp.raise_for_status()
        console.print(f"[green]✓ 知识库 {kb_id} 已清空[/green]")

    asyncio.run(_run())


@kb_app.command("delete")
def kb_delete(
    kb_id: str = typer.Option(..., "--kb-id", help="知识库 ID"),
    confirm: bool = typer.Option(False, "--yes", help="跳过确认"),
):
    """删除知识库（含所有向量、文档、会话）"""
    if not confirm:
        typer.confirm(f"确定要永久删除知识库 {kb_id}？", abort=True)

    async def _run():
        async with get_client() as client:
            resp = await client.delete(f"/knowledge-bases/{kb_id}")
            resp.raise_for_status()
        console.print(f"[green]✓ 知识库 {kb_id} 已删除[/green]")

    asyncio.run(_run())


# ========================
# 文档摄入
# ========================

@app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., help="文件路径（pdf/epub/md/txt）"),
    kb_id: str = typer.Option(..., "--kb-id", help="目标知识库 ID"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="是否等待摄入完成"),
):
    """上传并摄入文档到知识库"""
    if not path.exists():
        console.print(f"[red]文件不存在: {path}[/red]")
        raise typer.Exit(1)

    async def _run():
        async with get_client() as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"上传 {path.name}...", total=None)

                with open(path, "rb") as f:
                    resp = await client.post(
                        f"/knowledge-bases/{kb_id}/documents",
                        files={"file": (path.name, f, "application/octet-stream")},
                    )
                resp.raise_for_status()
                doc = resp.json()
                doc_id = doc["id"]

                progress.update(task, description=f"摄入中... (doc_id: {doc_id})")

                if wait:
                    while True:
                        status_resp = await client.get(
                            f"/knowledge-bases/{kb_id}/documents/{doc_id}/status"
                        )
                        status_data = status_resp.json()
                        cur_status = status_data["status"]

                        if cur_status == "indexed":
                            progress.stop()
                            console.print(Panel(
                                f"[green]✓ 摄入完成[/green]\n"
                                f"文档 ID: {doc_id}\n"
                                f"Chunk 数: {status_data['chunk_count']}\n"
                                f"图片数: {status_data['image_count']}",
                                title="摄入成功",
                            ))
                            break
                        elif cur_status == "failed":
                            progress.stop()
                            console.print(f"[red]✗ 摄入失败: {status_data.get('error_message')}[/red]")
                            raise typer.Exit(1)

                        await asyncio.sleep(2)
                else:
                    console.print(f"[yellow]文档已提交，后台处理中。doc_id: {doc_id}[/yellow]")

    asyncio.run(_run())


# ========================
# 检索
# ========================

@app.command("search")
def search(
    query: str = typer.Argument(..., help="检索查询"),
    kb_id: str = typer.Option(..., "--kb-id", help="知识库 ID"),
    top_k: int = typer.Option(5, "--top-k", help="返回结果数"),
):
    """执行混合检索（不生成回答）"""
    async def _run():
        async with get_client() as client:
            resp = await client.post(f"/knowledge-bases/{kb_id}/search", json={
                "query": query,
                "top_k": top_k,
            })
            resp.raise_for_status()
            data = resp.json()

        console.print(f"\n[bold]查询：[/bold] {query}\n")
        for i, result in enumerate(data["results"], 1):
            console.print(Panel(
                result["text"],
                title=f"[{i}] {result['source_doc']} "
                      f"(页:{result.get('page_num', '?')}) "
                      f"score={result['score']}",
                border_style="blue",
            ))

    asyncio.run(_run())


# ========================
# 对话
# ========================

@app.command("chat")
def chat(
    kb_id: str = typer.Option(..., "--kb-id", help="知识库 ID"),
    llm_provider: str = typer.Option("doubao", "--llm-provider", help="LLM 提供商"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM 模型 ID"),
):
    """交互式命令行对话"""
    console.print(Panel(
        f"[bold]BookRAG 对话模式[/bold]\n"
        f"知识库: {kb_id} | LLM: {llm_provider}\n"
        "输入 [bold]exit[/bold] 或 Ctrl+C 退出",
        title="对话",
    ))

    async def _create_session() -> str:
        from config import get_settings
        settings = get_settings()
        model = llm_model or {
            "doubao": settings.doubao_chat_model,
            "deepseek": settings.deepseek_model,
            "ollama": settings.ollama_default_chat_model,
        }.get(llm_provider, "default")

        async with get_client() as client:
            resp = await client.post("/sessions", json={
                "kb_id": kb_id,
                "llm_provider": llm_provider,
                "llm_model_id": model,
            })
            resp.raise_for_status()
            return resp.json()["id"]

    async def _stream_chat(session_id: str, query: str) -> str:
        full = []
        async with httpx.AsyncClient(
            base_url=DEFAULT_BASE_URL,
            timeout=300.0,
        ) as client:
            async with client.stream(
                "POST",
                f"/sessions/{session_id}/chat/stream",
                json={"content": query},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "token" in data:
                            token = data["token"]
                            full.append(token)
                            console.print(token, end="", highlight=False)
                        elif data.get("done"):
                            console.print()
                            sources = data.get("sources", [])
                            if sources:
                                console.print(f"\n[dim]来源: {', '.join(set(s['source_doc'] for s in sources[:3]))}[/dim]")
        return "".join(full)

    async def _main():
        session_id = await _create_session()
        console.print(f"[dim]会话 ID: {session_id}[/dim]\n")

        while True:
            try:
                query = input("\n[你] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]退出对话[/yellow]")
                break

            if query.lower() in ("exit", "quit", "q"):
                break
            if not query:
                continue

            console.print("\n[bold green]助手：[/bold green]", end="")
            await _stream_chat(session_id, query)

    asyncio.run(_main())


if __name__ == "__main__":
    app()
