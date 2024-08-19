import fnmatch
import logging
from typing import Optional, Type

from langchain.indexes import index
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.indexing import RecordManager
from langchain_text_splitters.base import TextSplitter
from watchdog.events import FileSystemEvent, FileSystemEventHandler


class IndexingEventHandler(FileSystemEventHandler):
    """A watchdog event handler that indexes files when they change."""

    def __init__(
        self,
        vector_store: VectorStore,
        record_manager: RecordManager,
        loader_cls: Type[TextLoader],
        splitter: TextSplitter,
        glob: str,
    ) -> None:
        super().__init__()
        self._vector_store = vector_store
        self._record_manager = record_manager
        self._loader_cls = loader_cls
        self._splitter = splitter
        self._glob = glob

    def _load_file(self, path: str) -> Optional[Document]:
        docs = self._loader_cls(path).load()
        if docs:
            return docs[0]
        return None

    def _delete_file(self, path: str) -> None:
        # The lanchain indexing API and utils don't have an elegant way to
        # clean up old documents. We're going to punch through the abstractions
        # until we throw them away.
        # tl;dr, we know the "group id" is a source id is a file path.
        uids_to_delete = self._record_manager.list_keys(group_ids=[path])
        if uids_to_delete:
            self._vector_store.delete(uids_to_delete)
            self._record_manager.delete_keys(uids_to_delete)
            logging.debug(f"Deleted {len(uids_to_delete)} records for {path}")

    def _index_file(self, path: str) -> None:
        doc = self._load_file(path)
        if doc:
            chunks = self._splitter.split_documents([doc])
            logging.debug(f"Indexing {len(chunks)} chunks from '{path}'")
            index(
                chunks,
                self._record_manager,
                self._vector_store,
                cleanup="incremental",
                source_id_key="source",
            )

    def on_moved(self, event: FileSystemEvent) -> None:
        """On file move, remove the old file from the index and add the new one."""
        # TODO: if a directory moves, do we get events for all the files in it?
        if event.is_directory:
            return
        if not fnmatch.fnmatch(event.dest_path, self._glob):
            return

        logging.debug(f"file move: from {event.src_path} to {event.dest_path}")
        self._delete_file(event.src_path)
        self._index_file(event.dest_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """On file creation, index the new file."""
        if not fnmatch.fnmatch(event.src_path, self._glob):
            return

        logging.debug(f"file created: {event.src_path}")
        self._index_file(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """On file deletion, remove the file from the index."""
        if not fnmatch.fnmatch(event.src_path, self._glob):
            return

        logging.debug(f"file deleted: {event.src_path}")
        self._delete_file(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """On file modification, re-index the file."""
        if not fnmatch.fnmatch(event.src_path, self._glob):
            return

        logging.debug(f"file modified: {event.src_path}")
        self._index_file(event.src_path)
