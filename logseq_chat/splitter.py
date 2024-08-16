from typing import Any

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
    """Splits Logseq data files into chunks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LogseqTextSplitter."""
        super().__init__(
            separators=SEPERATORS,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs,
        )
