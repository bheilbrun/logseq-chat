import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from logseq_chat.index import HybridSearchIndex
from logseq_chat.main import doc_id_func


def create_document(id: str, content: str, metadata: dict) -> Document:
    return Document(page_content=content, id=id, metadata=metadata)


@pytest.fixture
def hybrid_search_index() -> HybridSearchIndex:
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=2000))
    return HybridSearchIndex(vector_store, id_func=doc_id_func)


def test_add_documents(hybrid_search_index: HybridSearchIndex) -> None:
    docs = [
        create_document("1", "This is a test document", {"source": "file1.txt"}),
        create_document("2", "This is more content", {"source": "file1.txt"}),
    ]

    hybrid_search_index.add_documents(docs)

    assert len(hybrid_search_index.doc_store) == 2
    assert hybrid_search_index.bm25_index.corpus_size == 2


def test_add_duplicate_document(hybrid_search_index: HybridSearchIndex) -> None:
    doc = create_document("1", "This is a test document", {"source": "file1.txt"})

    hybrid_search_index.add_documents([doc])
    with pytest.raises(ValueError):
        hybrid_search_index.add_documents([doc])


def test_remove_documents_by_path(hybrid_search_index: HybridSearchIndex) -> None:
    docs = [
        create_document("1", "This is a test document", {"source": "file1.txt"}),
        create_document("2", "Another test document", {"source": "file2.txt"}),
    ]

    hybrid_search_index.add_documents(docs)
    hybrid_search_index.remove_documents_by_path("file1.txt")

    assert len(hybrid_search_index.doc_store) == 1
    assert hybrid_search_index.bm25_index.corpus_size == 1
    assert "1" not in hybrid_search_index.doc_store


def test_remove_nonexistent_document(hybrid_search_index: HybridSearchIndex) -> None:
    hybrid_search_index.remove_documents_by_path("nonexistent.txt")
    assert len(hybrid_search_index.doc_store) == 0


def test_search(hybrid_search_index: HybridSearchIndex) -> None:
    docs = [
        create_document(
            "1", "This is a test document about cats", {"source": "file1.txt"}
        ),
        create_document("2", "Another document about dogs", {"source": "file2.txt"}),
        create_document(
            "3",
            "A third document mentioning both cats and dogs",
            {"source": "file3.txt"},
        ),
    ]

    hybrid_search_index.add_documents(docs)
    results = hybrid_search_index.search("cats", k=2)

    assert len(results) == 2
    assert results[0].id in ["1", "3"]


def test_search_empty_index(hybrid_search_index: HybridSearchIndex) -> None:
    results = hybrid_search_index.search("test", k=5)
    assert len(results) == 0


def test_search_with_removed_document(hybrid_search_index: HybridSearchIndex) -> None:
    docs = [
        create_document("1", "This is a test document", {"source": "file1.txt"}),
        create_document("2", "Another test document", {"source": "file2.txt"}),
    ]

    hybrid_search_index.add_documents(docs)
    hybrid_search_index.remove_documents_by_path("file1.txt")
    results = hybrid_search_index.search("test", k=2)

    assert len(results) == 1
    assert results[0].id == "2"


def test_large_document_set(hybrid_search_index: HybridSearchIndex) -> None:
    # docs = [Document(page_content=f"Document {i}", id=str(i)) for i in range(1000)]
    docs = [
        create_document(str(i), f"Document {i}", {"source": f"file{i}.txt"})
        for i in range(1000)
    ]

    hybrid_search_index.add_documents(docs)

    assert hybrid_search_index.bm25_index.corpus_size == 1000
    results = hybrid_search_index.search("Document", k=10)
    assert len(results) == 10
