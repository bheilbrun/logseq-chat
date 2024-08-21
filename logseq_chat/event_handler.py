import fnmatch
import logging
from typing import Optional, Type

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from logseq_chat.index import HybridSearchIndex


class IndexingEventHandler(FileSystemEventHandler):
    """A watchdog event handler that indexes files when they change.

    Events like file modify and move are handled by calling delete_file and then
    add_file. There is no locking between these two operations currently, so
    there is a small window of time where the file is not indexed.
    """

    def __init__(
        self,
        search_index: HybridSearchIndex,
        loader_cls: Type[TextLoader],
        splitter: TextSplitter,
        glob: str,
    ) -> None:
        super().__init__()
        self._search_index = search_index
        self._loader_cls = loader_cls
        self._splitter = splitter
        self._glob = glob

    def _load_file(self, path: str) -> Optional[Document]:
        docs = self._loader_cls(path).load()
        if docs:
            return docs[0]
        return None

    def _delete_file(self, path: str) -> None:
        self._search_index.remove_documents_by_path(path)

    def _add_file(self, path: str) -> None:
        doc = self._load_file(path)
        if not doc:
            return
        chunks = self._splitter.split_documents([doc])
        self._search_index.add_documents(chunks)

    def on_moved(self, event: FileSystemEvent) -> None:
        """On file move, remove the old file from the index and add the new one."""
        # TODO: if a directory moves, do we get events for all the files in it?
        if event.is_directory:
            return
        if not fnmatch.fnmatch(event.dest_path, self._glob):
            return

        logging.debug(f"file move: from {event.src_path} to {event.dest_path}")
        self._delete_file(event.src_path)
        self._add_file(event.dest_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """On file creation, index the new file."""
        if not fnmatch.fnmatch(event.src_path, self._glob):
            return

        logging.debug(f"file created: {event.src_path}")
        self._add_file(event.src_path)

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
        self._add_file(event.src_path)
