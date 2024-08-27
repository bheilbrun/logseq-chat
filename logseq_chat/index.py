from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

from langchain_core.documents import Document

from logseq_chat.bm25 import IncrementalBM25
from logseq_chat.vector import VectorIndex


class SearchIndex(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def remove_documents_by_path(self, path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[Document]:
        pass


class HybridSearchIndex(SearchIndex):
    """Provides a hybrid search index that combines BM25 and embedding similarity.

    The API is optimized for the use case of indexing and searching chunks of
    documents. There may be multiple chunks for a given document, all of which
    would share the same underlying file path. Each chunk should however have
    a unique ID."""

    def __init__(
        self,
        vector_index: VectorIndex,
    ):
        self.doc_store: Dict[str, Document] = {}
        self.path_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)

        self.vector_index = vector_index
        # simple enough dep we don't need to use dependency injection.
        self.bm25_index = IncrementalBM25()

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Add documents to the index. Documents must have unique IDs set."""
        for doc in documents:
            id = doc.id
            source = doc.metadata.get("source")
            if not id or not source:
                raise ValueError(f"Document must have id and source set - {doc}")
            self.doc_store[id] = doc
            self.path_to_doc_ids[source].add(id)

        self.bm25_index.add_documents(documents)
        self.vector_index.add_documents(documents)

    def remove_documents_by_path(self, path: Union[str, Path]) -> None:
        """Remove documents with the given file paths from the index."""
        doc_ids = self.path_to_doc_ids.pop(str(path), set())
        if not doc_ids:
            return

        self.bm25_index.remove_documents(doc_ids)
        self.vector_index.remove_documents(list(doc_ids))
        for doc_id in doc_ids:
            self.doc_store.pop(doc_id)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the embedding and keyword indexes for the given query.
        The results of both backends are fused and ranked using Reciprocal Rank Fusion.
        """
        over_sampling = 2
        bm25_results = self.bm25_index.search(query, k * over_sampling)
        vector_results = self.vector_index.search(query, k * over_sampling)

        bm25_ids = [doc_id for doc_id, _ in bm25_results]
        vector_ids = [doc_id for doc_id, _ in vector_results]

        # Combine the results from both BM25 and vector similarity.
        fused_ranking = self._reciprocal_rank_fusion(bm25_ids, vector_ids, k)

        # Retrieve the documents from the docstore.
        results = [self.doc_store[doc_id] for doc_id, _ in fused_ranking]
        return results

    def _reciprocal_rank_fusion(
        self, id_list1: List[str], id_list2: List[str], k: int, c: int = 60
    ) -> List[Tuple[str, float]]:
        """Produce a single fused ranking of two ranked lists. RRF will boost documents
        that are ranked highly in both lists.

        The c paramater provides a smoothing factor to prevent overweighting the
        highest ranked documents."""
        rrf_scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(id_list1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + c)
        for rank, doc_id in enumerate(id_list2):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + c)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
