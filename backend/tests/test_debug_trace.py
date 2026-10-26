"""调试模式（RAG 全链路 trace）的采集内容与运行时开关。"""

from types import SimpleNamespace

import pytest

from core.observability import runtime as runtime_mod
from core.observability.trace import QueryTrace, new_trace, stage_metrics
from core.rag import pipeline as pipeline_mod
from core.rag.pipeline import RAGPipeline
from core.retrieval import retriever as retriever_mod
from core.retrieval.retriever import HybridRetriever
from vector_store.qdrant_store import SearchResult


def _result(chunk_id: str, score: float, text: str = "") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text or f"这是第{chunk_id}号片段，用来验证调试信息是否完整记录。",
        score=score,
        source_doc="book.md",
        doc_id="d1",
        chunk_type="text",
        page_num=3,
        metadata={"chunk_index": int(chunk_id), "doc_type": "md"},
    )


class _FakeEmbedding:
    async def embed_query(self, text, image=None):
        return [0.1, 0.2, 0.3]


class _ReverseReranker:
    model_name = "fake-reranker"

    async def rerank(self, query_text, results, top_k=None, trace=None):
        return list(reversed(results))


class _FakeLLM:
    config = SimpleNamespace(
        provider="fake",
        model_id="fake-1",
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
        reasoning_effort=None,
        timeout=30,
    )

    def __init__(self) -> None:
        self.messages = []

    async def stream_chat(self, messages):
        self.messages = messages
        for token in ["曾", "国藩"]:
            yield token


@pytest.fixture(autouse=True)
def _restore_runtime():
    runtime_mod.reset_runtime()
    yield
    runtime_mod.reset_runtime()


def test_runtime_switch_controls_trace_creation():
    runtime = runtime_mod.get_runtime()
    runtime.configure(enabled=True, layers=["retrieval"])

    trace = new_trace("q")
    assert trace is not None
    assert trace.on("retrieval") is True
    assert trace.on("query") is False

    runtime.configure(enabled=False)
    assert new_trace("q") is None


def test_runtime_rejects_unknown_layer():
    runtime = runtime_mod.get_runtime()
    snapshot = runtime.configure(enabled=True, layers=["retrieval", "not-a-layer"])
    assert snapshot["layers"] == ["retrieval"]


async def test_retrieval_trace_has_scores_metadata_and_drop_reason(monkeypatch, qdrant):
    monkeypatch.setattr(retriever_mod, "embed_sparse", lambda texts: [([1], [0.5])])
    qdrant.results["kb_x"] = [_result(str(i), score=1.0 - i * 0.1) for i in range(4)]

    trace = QueryTrace("question", layers="query,retrieval")
    retriever = HybridRetriever(qdrant, _FakeEmbedding(), _ReverseReranker())

    query = await retriever.build_query("question", top_k=2, score_threshold=0.75, trace=trace)
    out = await retriever.retrieve("kb_x", query, trace=trace)

    assert [r.chunk_id for r in out] == ["2", "1"]
    section = trace.sections["retrieval"]
    assert section["threshold"] == 0.75
    assert section["filtered_count"] == 1
    assert section["cache"]["enabled"] is False
    assert section["fusion"]["method"] == "rrf"
    assert section["fusion"]["k"] == 60
    assert section["fusion"]["prefetch_limit"] == section["params"]["prefetch_limit"]
    assert section["params"]["rerank_candidates"] == retriever_mod.settings.rerank_candidates

    rows = {row["chunk_id"]: row for row in section["results"]}
    kept = rows["2"]
    assert kept["included"] is True
    assert kept["final_rank"] == 1
    assert kept["vector_score"] is not None and kept["sparse_rank"] is not None
    assert kept["metadata"]["doc_type"] == "md"
    assert kept["page_num"] == 3
    assert rows["0"]["drop_reason"] == "beyond_top_k"
    assert rows["3"]["drop_reason"] == "below_score_threshold"


async def test_generation_trace_records_prompt_tokens_and_citations(monkeypatch, qdrant):
    monkeypatch.setattr(retriever_mod, "embed_sparse", lambda texts: [([1], [0.5])])
    monkeypatch.setattr(pipeline_mod.settings, "query_rewrite", "off")
    qdrant.results["kb_x"] = [_result("0", score=0.5), _result("1", score=0.4)]

    llm = _FakeLLM()
    retriever = HybridRetriever(qdrant, _FakeEmbedding())
    kb = SimpleNamespace(qdrant_collection_name="kb_x")
    pipeline = RAGPipeline(retriever=retriever, llm_provider=llm, kb=kb)

    trace = QueryTrace("question", layers="query,retrieval,generation,performance")
    events = [event async for event in pipeline.stream_chat("question", [], trace=trace)]

    assert [e["token"] for e in events if "token" in e] == ["曾", "国藩"]
    assert events[-1]["done"] is True

    query = trace.sections["query"]
    assert query["original"] == "question"
    assert query["rewrite"]["method"] == "none"
    assert query["retrieval_strategy"].startswith("hybrid")

    generation = trace.sections["generation"]
    assert generation["params"]["temperature"] == 0.2
    assert generation["prompt"]["context_ref_count"] == 2
    assert len(generation["prompt"]["messages"]) == 2  # system + user
    assert len(generation["context_chunks"]) == 2
    assert generation["tokens"]["prompt_estimated"] > 0
    assert generation["tokens"]["completion_estimated"] > 0
    assert generation["truncation"]["applied"] is False
    assert generation["output"] == "曾国藩"
    assert [c["cited"] for c in generation["citations"]] == [False, False]

    stages = {stage["name"] for stage in trace.stages}
    assert {"query_rewrite", "embedding", "qdrant_search", "prompt_build", "llm_generate"} <= stages


def test_stage_metrics_aggregates_percentiles():
    traces = [
        {"stages": [{"name": "retrieve", "duration_ms": 10}, {"name": "llm", "duration_ms": 100}]},
        {"stages": [{"name": "retrieve", "duration_ms": 30}]},
        {"stages": [{"name": "retrieve", "duration_ms": 20, "status": "error"}]},
    ]
    rows = {row["stage"]: row for row in stage_metrics(traces)}
    assert rows["retrieve"]["count"] == 3
    assert rows["retrieve"]["p50_ms"] == 20
    assert rows["retrieve"]["max_ms"] == 30
    assert rows["retrieve"]["errors"] == 1
