import math
from collections import Counter, defaultdict
from threading import Lock
from typing import Callable, Dict, Iterable, List, Tuple

from langchain_core.documents import Document


def default_tokenize_func(text: str) -> List[str]:
    """Super dumb tokenization function which lowers and splits on whitespace.
    TODO: make this better, e.g. stemming, stop word removal."""
    return text.lower().split()


class IncrementalBM25:
    """
    A thread-safe BM25 Okapi implementation which supports incremental updates.

    Adding and removing documents is O(n) in the number of terms in the document.

    Search is O(n) where n in the number of terms in the query. If any documents have
    changed since the last search, it is O(n+m) where m is the number of terms in the
    corpus. This is because the BM25 IDF values are updated at search time.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenize_func: Callable[[str], List[str]] = default_tokenize_func,
    ) -> None:
        # BM25 parameters - https://en.wikipedia.org/wiki/Okapi_BM25
        self.k1 = k1
        self.b = b

        self.tokenize_func = tokenize_func

        # Ensure thread safety of below internal state.
        self._lock = Lock()
        # Lazy update of IDF values at search-time if necessary.
        self._dirty = False

        # BM25 statistics
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0
        self.doc_freqs: Counter[str] = Counter()  # term to document frequency
        self.idf: Dict[str, float] = {}

        # Corpus statistics.
        self.doc_len: Dict[str, int] = {}  # id to doc length
        self.doc_term_freqs: Dict[str, Counter] = {}  # id to term frequencies

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Add documents to the index. Documents must have unique IDs set."""
        if not documents:
            return

        with self._lock:
            self._dirty = True
            for doc in documents:
                self._add_document(doc)

    def _add_document(self, doc: Document) -> None:
        """Add a document to the index. Must be called with lock acquired."""
        if not doc.id:
            raise ValueError(
                f"Document must have an id. Source: {doc.metadata.get('source')}"
            )
        if doc.id in self.doc_len:
            raise ValueError(f"Document with id {doc.id} already exists.")

        # TODO: Index metadata as well.
        tokenized_doc = self.tokenize_func(doc.page_content)
        doc_len = len(tokenized_doc)
        term_freqs = Counter(tokenized_doc)

        self.doc_len[doc.id] = doc_len
        self.doc_term_freqs[doc.id] = term_freqs
        self.corpus_size += 1

        for term, _ in term_freqs.items():
            self.doc_freqs[term] += 1

        self.avg_doc_len = sum(self.doc_len.values()) / self.corpus_size

    def remove_documents(self, doc_ids: Iterable[str]) -> None:
        """Remove documents from the index."""
        if not doc_ids:
            return

        with self._lock:
            self._dirty = True
            for doc_id in doc_ids:
                self._remove_document(doc_id)

    def _remove_document(self, doc_id: str) -> None:
        """Remove a document from the index. Must be called with lock acquired."""
        if not doc_id:
            raise ValueError("Document id must not be provided and non-empty.")
        if doc_id not in self.doc_len:
            raise ValueError(f"Document with id {doc_id} does not exist.")

        term_freqs = self.doc_term_freqs.pop(doc_id)
        self.doc_len.pop(doc_id)
        self.corpus_size -= 1

        for term, _ in term_freqs.items():
            self.doc_freqs[term] -= 1
            if self.doc_freqs[term] == 0:
                del self.doc_freqs[term]

        if self.corpus_size > 0:
            self.avg_doc_len = sum(self.doc_len.values()) / self.corpus_size
        else:
            self.avg_doc_len = 0

    def _update_idf(self) -> None:
        """Updates the Inverse Document Frequency (IDF) values for all terms in
        the index. Must be called with self._lock acquired."""
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(
                1 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )
        self._dirty = False

    def _get_scores(self, query: str) -> List[Tuple[str, float]]:
        """Returns the score for every document in the index for the given query.
        Must be called with self._lock acquired."""
        query_terms = self.tokenize_func(query)
        scores: Dict[str, float] = defaultdict(float)
        for q in query_terms:
            q_idf = self.idf.get(q, 0)
            if q_idf == 0:
                # skip terms not in the corpus
                continue
            for doc_id, term_freqs in self.doc_term_freqs.items():
                if q not in term_freqs:
                    continue
                score = self._score(q_idf, term_freqs[q], self.doc_len[doc_id])
                scores[doc_id] += score
        return list(scores.items())

    def _score(self, idf: float, freq: int, doc_len: int) -> float:
        """Calculate a single score."""
        num = idf * freq * (self.k1 + 1)
        denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
        return num / denom

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """For the given query, returns the top k document IDs and scores.
        Terms which are not in the corpus are ignored."""
        with self._lock:
            # Lazy update IDF values (expensive) if docs were removed/added.
            if self._dirty:
                self._update_idf()
            scores = self._get_scores(query)
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
