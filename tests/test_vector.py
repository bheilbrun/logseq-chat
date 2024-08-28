import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document

from logseq_chat.vector import SQLiteVecVectorIndex


@pytest.fixture
def vector_index() -> SQLiteVecVectorIndex:
    return SQLiteVecVectorIndex(
        namespace="test",
        embedding=DeterministicFakeEmbedding(size=16),
        embedding_dim=16,
    )


def test_add_documents(vector_index: SQLiteVecVectorIndex) -> None:
    docs = [
        Document(page_content="apple", id="a"),
        Document(page_content="banana", id="b"),
        Document(page_content="cherry", id="c"),
    ]

    vector_index.add_documents(docs)

    results = vector_index.search("apple", k=3)
    assert len(results) == 3
    assert results[0][0] == "a"


def test_remove_documents(vector_index: SQLiteVecVectorIndex) -> None:
    docs = [
        Document(page_content="dog", id="d"),
        Document(page_content="elephant", id="e"),
    ]

    vector_index.add_documents(docs)
    vector_index.remove_documents(["d"])

    results = vector_index.search("dog", k=2)
    assert len(results) == 1
    assert results[0][0] == "e"


def test_search(vector_index: SQLiteVecVectorIndex) -> None:
    docs = [
        Document(page_content="red", id="r"),
        Document(page_content="green", id="g"),
        Document(page_content="blue", id="b"),
    ]
    vector_index.add_documents(docs)

    results = vector_index.search("red", k=2)
    assert len(results) == 2
    assert results[0][0] == "r"
    assert results[1][0] in ["g", "b"]


def test_add_existing_document(vector_index: SQLiteVecVectorIndex) -> None:
    doc = Document(page_content="existing", id="9")

    vector_index.add_documents([doc])
    vector_index.add_documents([doc])

    results = vector_index.search("existing", k=2)
    assert len(results) == 1
    assert results[0][0] == "9"


def test_remove_nonexistent_document(vector_index: SQLiteVecVectorIndex) -> None:
    # This should not raise an error
    vector_index.remove_documents(["nonexistent_id"])


def test_search_empty_index(vector_index: SQLiteVecVectorIndex) -> None:
    results = vector_index.search("query", k=5)
    assert len(results) == 0
