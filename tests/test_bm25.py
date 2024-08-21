import pytest
from langchain_core.documents import Document

from logseq_chat.bm25 import IncrementalBM25


@pytest.fixture
def bm25_instance() -> IncrementalBM25:
    return IncrementalBM25()


def test_add_documents(bm25_instance: IncrementalBM25) -> None:
    doc = Document(page_content="This is a test document", id="1")

    bm25_instance.add_documents([doc])

    assert len(bm25_instance.doc_len) == 1
    assert bm25_instance._dirty


def test_remove_documents(bm25_instance: IncrementalBM25) -> None:
    doc = Document(page_content="This is a test document", id="1")

    bm25_instance.add_documents([doc])
    bm25_instance.remove_documents(["1"])

    assert len(bm25_instance.doc_len) == 0
    assert bm25_instance._dirty


def test_search(bm25_instance: IncrementalBM25) -> None:
    doc1 = Document(page_content="This is a test document", id="1")
    doc2 = Document(page_content="Another document for testing", id="2")

    bm25_instance.add_documents([doc1])
    bm25_instance.add_documents([doc2])
    results = bm25_instance.search("test document", k=1)

    assert len(results) == 1
    id, score = results[0]
    assert id == "1"
    assert score > 0
    assert not bm25_instance._dirty


def test_non_existent_search(bm25_instance: IncrementalBM25) -> None:
    doc = Document(page_content="This is a test document", id="1")

    bm25_instance.add_documents([doc])
    results = bm25_instance.search("nonexistent", k=1)

    assert len(results) == 0
