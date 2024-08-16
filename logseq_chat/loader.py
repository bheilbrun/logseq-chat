import logging
from typing import Iterator, Union

from click import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class LogseqMarkdownLoader(TextLoader):
    """
    Loads a single Logseq Markdown file.

    If the file contains properties of the form "foo::bar", these will be inserted
    into the document's metadata.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
    ):
        super().__init__(file_path)  # type: ignore

    def lazy_load(self) -> Iterator[Document]:
        """Load from file path."""
        docs = super().lazy_load()
        for doc in docs:
            # find lines of the form "foo::bar" and inserts them into doc.metadata
            for line in doc.page_content.splitlines():
                if "::" not in line:
                    # properties are at the top of the file.
                    # break if we reach a non-property line
                    break

                key, value = line.split("::", 1)
                if key in doc.metadata:
                    logging.debug(
                        "Not overwriting existing metadata key '%s' for '%s' ",
                        key,
                        doc.metadata.get("source", ""),
                    )
                    continue

                doc.metadata[key] = value.strip()
                continue

            yield doc
