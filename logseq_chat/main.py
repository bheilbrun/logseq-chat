import logging
import os
import textwrap
from hashlib import sha256
from typing import List

import click
import faiss
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_debug, set_verbose
from langchain.storage import LocalFileStore
from langchain_anthropic import ChatAnthropic
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TextSplitter
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from logseq_chat.event_handler import IndexingEventHandler
from logseq_chat.index import HybridSearchIndex
from logseq_chat.loader import LogseqMarkdownLoader
from logseq_chat.splitter import LogseqMarkdownSplitter

_verbose = False
_debug = False


def set_verbosity(verbose: bool, debug: bool) -> None:
    global _verbose, _debug
    _verbose = verbose
    _debug = debug
    set_verbose(verbose)
    set_debug(debug)
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    # low signal loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_dir(data_dir: str, glob: str) -> List[Document]:
    """
    Load Logseq data from the given directory. Returns list of Langchain Documents.
    """
    loader = DirectoryLoader(
        data_dir,
        glob=glob,
        loader_cls=LogseqMarkdownLoader,
        show_progress=_verbose,
    )
    docs = loader.load()
    logging.debug("First doc: %s", textwrap.shorten(str(docs[0]), width=72))
    return docs


def doc_id_func(doc: Document) -> str:
    """Hash a document. Excludes the id field to allow for stable id generation.
    This should be a feature of the Document class, but it's not. If it's
    replaced with a local abstraction that can be fixed."""
    return sha256(doc.json(exclude={"id"}).encode()).hexdigest()


# Default chunk size for splitting Logseq data.
# Measured in "characters" as defined by Python len() function.
# Note: there are roughly 4 chars per token in English text.
DEFAULT_CHUNK_SIZE_CHARS = 4096


def split_docs(splitter: TextSplitter, docs: List[Document]) -> List[Document]:
    """Split a list of documents into chunks."""
    chunks = splitter.split_documents(docs)
    logging.debug("First chunk: %s", textwrap.shorten(str(chunks[0]), width=72))
    return chunks


MODEL_NAME = "text-embedding-3-small"
MODEL_DIM = 1536


def get_vector_store(
    model_name: str = MODEL_NAME, model_dim: int = MODEL_DIM
) -> VectorStore:
    embeddings_model = OpenAIEmbeddings(model=model_name)
    store = LocalFileStore(".cache")
    # Caches document embeddings to disk. Does not cache queries.
    # TODO: inefficient and eternal. Replace.
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_model,
        store,
        namespace=embeddings_model.model,
    )
    # TODO: Add metadata to the embedded content. FAISS only embeds the page_content.
    vector_store = FAISS(
        cached_embedder,
        index=faiss.IndexFlatL2(model_dim),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vector_store


def get_llm() -> BaseChatModel:
    """Checks for API keys in the environment and returns an appropriate LLM."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Chatting with Anthropic Claude 3.5 Sonnet...")
        return ChatAnthropic(
            model_name="claude-3-5-sonnet-20240620",
            temperature=0.0,
        )  # type: ignore
    if os.environ.get("OPENAI_API_KEY"):
        print("Chatting with OpenAI GPT 4o Mini...")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    raise ValueError("No API key found for OpenAI or Anthropic")


def schedule_observer(
    search_index: HybridSearchIndex,
    splitter: TextSplitter,
    data_dir: str,
    glob: str,
) -> BaseObserver:
    """Starts and returns a watchdog Observer which watches for file changes."""
    event_handler = IndexingEventHandler(
        search_index,
        loader_cls=LogseqMarkdownLoader,
        splitter=splitter,
        glob=glob,
    )
    observer = Observer()
    print(f"Watching directory: {data_dir}")
    observer.schedule(event_handler, data_dir, recursive=True)
    observer.start()
    return observer


def _format_docs(docs: List[Document]) -> str:
    """Helper to format Documents for the LLM prompt."""
    context = ""
    for doc in docs:
        title = doc.metadata.get("properties", {}).get("title", None)
        source = doc.metadata.get("source", None)
        if source:
            source = os.path.splitext(os.path.basename(source))[0]
        context += (
            "----\n"
            f"DOCUMENT TITLE: {title or source or 'Untitled'}\n"
            f"{doc.page_content}\n"
            "----\n"
        )
    return context


# Examples,
# Langchain RAG tutorial,
# https://python.langchain.com/v0.2/docs/tutorials/rag
# Anthropic prompt library "Cite your sources",
# https://docs.anthropic.com/en/prompt-library/cite-your-sources
SYSTEM_PROMPT = (
    "You are an expert assistant for question-answering tasks. "
    "Use the chat history and the following context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    'After answering, add a "Pages:" section with a bulleted list '
    "of any cited document titles. Titles should be in the format "
    "[[Document Title]]."
    "\n\n"
    "CONTEXT:\n"
    "{context}"
)

QUERY_REWRITE_SYSTEM_PROMPT = """\
You are a helpful assistant that rewrites user questions to optimize for document retrieval. Your task:

1. Analyze the user's question and chat history.
2. Rewrite the question to be self-contained and context-aware.
3. Optimize for similarity search:
   - Use clear, concise language
   - Include relevant keywords
   - Expand acronyms when needed
4. Maintain the original intent and meaning.
5. Don't add information not implied by the question or chat history.

Provide only the rewritten question, without explanations.

Example:
Question: "What were its key features?"
History: 
- User: "Tell me about the iPhone 12."
- Assistant: "The iPhone 12 was released in 2020..."
- User: "What were its key features?"

Rewritten: "What were the key features of the iPhone 12 released in 2020?"
"""

# TODO: make this configurable.
# TODO: consider summarizing the chat history when over the message limit.
CHAT_HISTORY_MESSAGES_LIMIT = 10
assert (
    CHAT_HISTORY_MESSAGES_LIMIT % 2 == 0
), "Langchain's Anthropic support breaks down if given an odd number of messages"

# Number of documents to retrieve and send to the LLM.
RETRIEVAL_TOP_K = 10


def interactive_loop(search_index: HybridSearchIndex) -> None:
    """Runs the interactive REPL loop until the user exits."""
    llm = get_llm()
    chat_history_store = InMemoryChatMessageHistory()

    # Chain to rewrite user's query for improved document retrieval performance.
    # TODO: use cheaper/faster LLM for query rewrites vs chat.
    query_rewrite_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_REWRITE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # Chain to answer the user's query using chat history and retrieved documents.
    chat_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    click.echo("Enter 'exit', 'quit' or ctrl-d to quit.")
    while True:
        query = click.prompt("", prompt_suffix="> ")
        if not query:
            continue
        if query.strip().lower() in ("exit", "quit"):
            break

        chat_history = chat_history_store.messages[:CHAT_HISTORY_MESSAGES_LIMIT]
        retrieval_query = query_rewrite_chain.invoke(
            {"input": query, "chat_history": chat_history}
        )
        logging.debug("Rewritten retrieval query: %s", retrieval_query)
        docs = search_index.search(retrieval_query, RETRIEVAL_TOP_K)

        resp_iter = chat_chain.stream(
            {
                "context": _format_docs(docs),
                "chat_history": chat_history,
                "input": query,
            }
        )
        full_resp = ""
        for chunk in resp_iter:
            print(chunk, end="", flush=True)
            full_resp += chunk
        print()

        # Only record the user query if we also have an AI response.
        # Anthropic requires alternating user/AI messages.
        chat_history_store.add_user_message(query)
        chat_history_store.add_ai_message(full_resp)


@click.command()
@click.option(
    "--data-dir", default=".", help="Directory containing Logseq data (markdown files)"
)
@click.option("--glob", default="**/*.md", help="Glob pattern for files to load")
@click.option("--verbose", is_flag=True, default=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def main(data_dir: str, glob: str, verbose: bool, debug: bool) -> None:
    load_dotenv()
    set_verbosity(verbose, debug)
    vector_store = get_vector_store()
    search_index = HybridSearchIndex(vector_store, doc_id_func)
    # record_manager = get_record_manager(vector_store)
    splitter = LogseqMarkdownSplitter(
        id_func=doc_id_func, chunk_size=DEFAULT_CHUNK_SIZE_CHARS, chunk_overlap=0
    )
    observer = schedule_observer(
        search_index,
        splitter,
        data_dir,
        glob,
    )
    try:
        docs = load_dir(data_dir, glob)
        print(f"Loaded {len(docs)} docs")
        chunks = split_docs(splitter, docs)
        print(f"Split into {len(chunks)} chunks")
        search_index.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks")

        # enter the main repl loop and block until the user exits
        interactive_loop(search_index)
    except (KeyboardInterrupt, click.exceptions.Abort):
        print("Exiting...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
