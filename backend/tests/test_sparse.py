"""BM25 稀疏编码：中文分词是否可用、是否稳定。"""

from core.retrieval.sparse import TOKENIZER_VERSION, encode, embed_sparse, tokenize


def test_chinese_query_is_split_into_multiple_terms():
    tokens = tokenize("李鸿章和曾国藩的关系")
    assert len(tokens) > 3
    # 单字与二元组都要有，整句不应被当成一个 token
    assert "李鸿" in tokens
    assert "鸿章" in tokens
    assert "国藩" in tokens
    assert "李鸿章和曾国藩的关系" not in tokens
    # 高频虚词单字被过滤
    assert "的" not in tokens


def test_long_sentence_is_not_a_single_term():
    text = "因此在镇压太平天国、兴起洋务运动的同时，曾国藩还致力于培养正人"
    tokens = tokenize(text)
    assert len(tokens) > 20
    assert all(len(token) <= 40 for token in tokens)


def test_document_and_query_share_terms():
    doc = set(tokenize("李鸿章是曾国藩的学生，二人关系密切。"))
    query = set(tokenize("李鸿章和曾国藩是什么关系？"))
    # 查询里的专名二元组必须能在文档里命中，这是 BM25 有结果的前提
    assert {"李鸿", "鸿章", "国藩"} <= doc & query


def test_english_words_are_lowercased_and_stopwords_dropped():
    tokens = tokenize("The RAG Pipeline uses Qdrant for RRF fusion")
    assert "rag" in tokens
    assert "qdrant" in tokens
    assert "the" not in tokens
    assert "for" not in tokens


def test_encode_is_deterministic_and_sorted():
    first = encode("曾国藩与李鸿章共同剿捻")
    second = encode("曾国藩与李鸿章共同剿捻")
    assert first == second
    indices, values = first
    assert indices == sorted(indices)
    assert len(indices) == len(values) > 0
    assert all(0 <= index < 2**32 for index in indices)
    assert all(value > 0 for value in values)


def test_encode_handles_empty_text():
    assert encode("") == ([], [])
    assert embed_sparse([]) == []


def test_tokenizer_version_is_exposed():
    # 版本号用于提示「改了分词就要重建索引」
    assert TOKENIZER_VERSION >= 2


def test_embed_sparse_keeps_input_order():
    results = embed_sparse(["曾国藩", "李鸿章", "完全无关的一句话"])
    assert len(results) == 3
    assert results[0] != results[1]
