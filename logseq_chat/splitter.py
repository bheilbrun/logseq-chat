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


# Default chunk size for splitting Logseq data.
# Measured in "characters" as defined by Python len() function.
# Note: there are roughly 4 chars per token in English text.
DEFAULT_CHUNK_SIZE_CHARS = 4096


class LogseqMarkdownSplitter(RecursiveCharacterTextSplitter):
    """Splits Logseq markdown files into chunks."""

    def __init__(
        self,
        id_func: Callable[[Document], str],
        chunk_size: int = DEFAULT_CHUNK_SIZE_CHARS,
        chunk_overlap: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LogseqTextSplitter.

        id_func is a function that takes a Document and returns a unique ID for that
        document. This should be a feature of the Document class, but it's not."""

        self.id_func = id_func
        super().__init__(
            separators=SEPERATORS,
            is_separator_regex=True,
            keep_separator=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
