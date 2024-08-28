import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Tuple

import sqlite_vec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorIndex(ABC):
    @abstractmethod
    def add_documents(self, documents: Iterable[Document]) -> None:
        pass

    @abstractmethod
    def remove_documents(self, ids: Iterable[str]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[Tuple[str, Any]]:
        pass


class SQLiteVecVectorIndex(VectorIndex):
    """
    A simple vector search index based on Sqlite with the sqlite_vec extension.

    Stores only document IDs and embeddings. Document contents must be stored externally.
    In this way, we avoid storing document contents in multiple places.
    """

    def __init__(
        self,
        namespace: str,
        embedding: Embeddings,
        embedding_dim: int,
        db_path: str = ":memory:",
    ) -> None:
        """Initialize the SQLiteVecVectorStore.

        The 'namespace' should include the name of the model used to generate the
        embeddings so that searches are guaranteed to be in the same embedding space.

        It'd be nice if the Embeddings abstraction exposed the model name and number
        of dimensions, but it doesn't. A future rewrite should include these.
        """
        self.embedding = embedding
        self.embedding_dim: int = embedding_dim
        self.namespace = namespace.replace(" ", "_").replace("-", "_")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.create_namespace_if_not_exists()

    def create_namespace_if_not_exists(self) -> None:
        """
        Create the necessary tables for this namespace if they don't already exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.namespace}_metadata (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL UNIQUE
            )
        """
        )
        # Index for fast lookups by doc_id
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.namespace}_doc_id 
            ON {self.namespace}_metadata(doc_id)
        """
        )
        # Cosine is slightly cheaper to compute than L2 distance.
        # OpenAI and Voyage say that L2 and Cosine rankings will be equivalent.
        cursor.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.namespace}_embeddings 
            USING vec0(embedding FLOAT[{self.embedding_dim}] distance_metric=cosine)
        """
        )
        self.conn.commit()

    def add_documents(self, documents: Iterable[Document]) -> None:
        """
        Add documents to the vector store.

        If a document with the same ID already exists, it will not be re-added.
        """
        cursor = self.conn.cursor()

        # First, filter out documents that already exist
        new_docs = []
        for doc in documents:
            doc_id = doc.id
            if not doc_id:
                raise ValueError("Document must have an ID set")
            # TODO: optimize by batching the query
            cursor.execute(
                f"SELECT 1 FROM {self.namespace}_metadata WHERE doc_id = ?",
                (doc_id,),
            )
            if cursor.fetchone() is None:
                new_docs.append((doc, doc_id))

        if not new_docs:
            return  # All documents already exist, no need to do anything

        # Generate embeddings only for new documents
        # TODO: include metadata in the embedded content
        embeddings = self.embedding.embed_documents(
            [doc.page_content for doc, _ in new_docs]
        )

        for (doc, doc_id), embedding in zip(new_docs, embeddings):
            cursor.execute(
                f"""
                INSERT INTO {self.namespace}_metadata (doc_id) 
                VALUES (?)
            """,
                (doc_id,),
            )
            vector_id = cursor.lastrowid
            cursor.execute(
                f"""
                INSERT INTO {self.namespace}_embeddings (rowid, embedding) 
                VALUES (?, ?)
            """,
                (vector_id, sqlite_vec.serialize_float32(embedding)),
            )

        self.conn.commit()

    def remove_documents(self, ids: Iterable[str]) -> None:
        """Remove documents from the vector store. Non-existent IDs are ignored."""
        cursor = self.conn.cursor()
        # TODO: rewrite as a batch query
        for doc_id in ids:
            cursor.execute(
                f"SELECT vector_id FROM {self.namespace}_metadata WHERE doc_id = ?",
                (doc_id,),
            )
            result = cursor.fetchone()
            if result:
                vector_id = result[0]
                cursor.execute(
                    f"DELETE FROM {self.namespace}_metadata WHERE doc_id = ?", (doc_id,)
                )
                cursor.execute(
                    f"DELETE FROM {self.namespace}_embeddings WHERE rowid = ?",
                    (vector_id,),
                )
        self.conn.commit()

    def search(self, query: str, k: int = 4) -> List[Tuple[str, Any]]:
        """
        Search for similar documents in the vector store.
        Returns a list of (id, distance) pairs.
        """
        query_embedding = self.embedding.embed_query(query)
        cursor = self.conn.cursor()
        results = cursor.execute(
            f"""
            SELECT 
                m.doc_id,
                e.distance
            FROM {self.namespace}_embeddings e
            JOIN {self.namespace}_metadata m ON e.rowid = m.vector_id
            WHERE
                e.embedding MATCH ?
                AND k = ?
            ORDER BY e.distance
        """,
            (sqlite_vec.serialize_float32(query_embedding), k),
        ).fetchall()

        return [(row[0], row[1]) for row in results]

    def __del__(self) -> None:
        if hasattr(self, "conn"):
            self.conn.close()
