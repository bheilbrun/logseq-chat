import math
import re
from collections import Counter, defaultdict
from threading import Lock
from typing import Dict, Iterable, List, Tuple

from langchain_core.documents import Document


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
    ) -> None:
        # BM25 parameters - https://en.wikipedia.org/wiki/Okapi_BM25
        self.k1 = k1
        self.b = b

        self.tokenizer: Tokenizer = Tokenizer()

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
        tokenized_doc = self.tokenizer.tokenize(doc.page_content)
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
        query_terms = self.tokenizer.tokenize(query)
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


class Tokenizer:
    """A simple English-focused tokenizer. Its only dependency is the re module."""

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess the input text using a language-agnostic approach.

        Steps:
        1. Convert to lowercase
        2. Split on non-alphanumeric characters (e.g. spaces, punctuation)
        4. Remove stop words and very short tokens

        Args:
        text (str): The input text to tokenize and preprocess.

        Returns:
        List[str]: A list of preprocessed tokens.
        """
        # Convert to lowercase
        text = text.lower()

        tokens = re.split(r"\W+", text)

        # Remove stop words and very short tokens
        tokens = [
            token for token in tokens if token not in STOP_WORDS and len(token) > 1
        ]

        return tokens


# Copied from spaCy's set of English stop words.
# https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py
# This version doesn't handle apostrophes or contractions.
STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split()
)
