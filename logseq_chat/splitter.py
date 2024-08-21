from typing import Any, Callable, Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# Logseq markdown files are primarly composed of bulleted lists
# with zero or more leading tabs. A "page" (as opposed to a journal) may
# begin with metadata fields of the form "foo:: bar" followed by a newline.
#
# By using '\n' as the last available separator, a single very long line
# could exceed the maximum chunk size. Consider adding " " and "" as
# additional separators.
SEPERATORS = [
    "\t{0,99}- ",
    "\n\n",
    "\n",
]


class LogseqMarkdownSplitter(RecursiveCharacterTextSplitter):
    """Splits Logseq markdown files into chunks."""

    def __init__(self, id_func: Callable[[Document], str], **kwargs: Any) -> None:
        """Initialize a LogseqTextSplitter.

        id_func is a function that takes a Document and returns a unique ID for that
        document. This should be a feature of the Document class, but it's not."""

        self.id_func = id_func
        super().__init__(
            separators=SEPERATORS,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs,
        )

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Split documents into chunks of the configured max size. The id field of each
        returned chunk will be set to a hash of the chunk's content and metadata.
        """
        chunks = super().split_documents(documents)
        for chunk in chunks:
            chunk.id = self.id_func(chunk)
        return chunks
