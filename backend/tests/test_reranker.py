"""Rerank：HTTP 客户端、失败降级与检索链路的候选池/截断行为。"""

import json

import httpx

from core.retrieval import retriever as retriever_mod
from core.retrieval.reranker import HttpReranker, _parse_rerank_scores
from core.retrieval.retriever import HybridRetriever
from vector_store.qdrant_store import SearchResult


def _result(chunk_id: str, score: float = 0.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        score=score,
        source_doc="a.md",
        doc_id="d1",
        chunk_type="text",
    )


class _FakeEmbedding:
    async def embed_query(self, text, image=None):
        return [0.1, 0.2, 0.3]


class _RecordingReranker:
    """记录收到的候选，并反转顺序，便于确认结果确实被重排"""

    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query_text, results, top_k=None):
        self.calls.append((query_text, [r.chunk_id for r in results]))
        ordered = list(reversed(results))
        return ordered[:top_k] if top_k else ordered


def test_parse_rerank_scores_backfills_by_index():
    body = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
            {"index": 9, "relevance_score": 1.0},  # 越界，忽略
            {"no_index": True},  # 缺 index，忽略
        ]
    }
    assert _parse_rerank_scores(body, 3) == [0.1, 0.0, 0.9]


async def test_http_reranker_orders_by_score_and_truncates():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "BAAI/bge-reranker-v2-m3"
        assert payload["documents"] == ["text-a", "text-b", "text-c"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = HttpReranker(
        "http://rerank.local/v1/", "BAAI/bge-reranker-v2-m3", client=client
    )
    try:
        out = await reranker.rerank(
            "question",
            [_result("a", 0.5), _result("b", 0.4), _result("c", 0.3)],
            top_k=2,
        )
    finally:
        await client.aclose()

    assert [r.chunk_id for r in out] == ["b", "c"]
    # score 保留融合分，rerank 分单独带出（接口用 rerank_score 展示）
    assert [r.score for r in out] == [0.4, 0.3]
    assert [r.rerank_score for r in out] == [0.9, 0.5]


async def test_http_reranker_falls_back_and_cools_down():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = HttpReranker("http://rerank.local/v1", "m", client=client)
    results = [_result("a", score=0.1), _result("b", score=0.2)]
    try:
        first = await reranker.rerank("q", results, top_k=1)
        second = await reranker.rerank("q", results, top_k=1)
    finally:
        await client.aclose()

    # 失败时保持原顺序并截断；熔断期内不再发请求
    assert [r.chunk_id for r in first] == ["a"]
    assert [r.chunk_id for r in second] == ["a"]
    assert len(calls) == 1


async def test_retrieve_expands_candidates_then_truncates(monkeypatch, qdrant):
    monkeypatch.setattr(retriever_mod, "embed_sparse", lambda texts: [([1], [0.5])])
    monkeypatch.setattr(retriever_mod.settings, "rerank_candidates", 5)

    qdrant.results["kb_x"] = [_result(str(i), score=1.0 - i * 0.1) for i in range(5)]
    reranker = _RecordingReranker()
    retriever = HybridRetriever(qdrant, _FakeEmbedding(), reranker)

    query = await retriever.build_query("question", top_k=2)
    out = await retriever.retrieve("kb_x", query)

    # 召回放大到 rerank_candidates，prefetch 也相应放大；最终仍截断到 top_k
    assert qdrant.last_search["top_k"] == 5
    assert qdrant.last_search["prefetch_limit"] == 20  # max(20, top_k * 2)
    assert reranker.calls == [("question", ["0", "1", "2", "3", "4"])]
    assert [r.chunk_id for r in out] == ["4", "3"]


async def test_retrieve_skips_reranker_when_disabled(monkeypatch, qdrant):
    monkeypatch.setattr(retriever_mod, "embed_sparse", lambda texts: [([1], [0.5])])

    qdrant.results["kb_x"] = [_result(str(i), score=1.0 - i * 0.1) for i in range(5)]
    reranker = _RecordingReranker()
    retriever = HybridRetriever(qdrant, _FakeEmbedding(), reranker)

    query = await retriever.build_query("question", top_k=2)
    out = await retriever.retrieve("kb_x", query, use_reranker=False)

    assert qdrant.last_search["top_k"] == 2
    assert reranker.calls == []
    assert [r.chunk_id for r in out] == ["0", "1"]
