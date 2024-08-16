import logging
import os
import textwrap
from typing import List

import click
import faiss
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_debug, set_verbose
from langchain.indexes import SQLRecordManager, index
from langchain.storage import LocalFileStore
from langchain_anthropic import ChatAnthropic
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.indexing import RecordManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TextSplitter
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from logseq_chat.event_handler import IndexingEventHandler
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


def load_dir(data_dir: str) -> List[Document]:
    """
    Load Logseq data from the given directory. Returns list of Langchain Documents.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/H100 Testing.md",
        loader_cls=LogseqMarkdownLoader,
        show_progress=_verbose,
    )
    docs = loader.load()
    logging.debug("First doc: %s", textwrap.shorten(str(docs[0]), width=72))
    return docs


# Default chunk size for splitting Logseq data as measured by Python len() function.
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


RECORD_MANAGER_CACHE_FILE: str = ".cache/record_manager_cache.sql"


def get_record_manager(vector_store: VectorStore) -> RecordManager:
    """
    Returns a RecordManager for the given VectorStore.

    This is Langchain's API for indexing.

    TODO: replace due to its limitations.
    """
    namespace = f"{vector_store.__class__.__name__}/{MODEL_NAME}/logseq"
    # Start from scratch each time to avoid out-of-sync issues with the vector store.
    if os.path.exists(RECORD_MANAGER_CACHE_FILE):
        os.remove(RECORD_MANAGER_CACHE_FILE)
    record_manager = SQLRecordManager(
        namespace, db_url=f"sqlite:///{RECORD_MANAGER_CACHE_FILE}"
    )
    record_manager.create_schema()
    return record_manager


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
    vector_store: VectorStore,
    record_manager: RecordManager,
    splitter: TextSplitter,
    data_dir: str,
) -> BaseObserver:
    """Starts and returns a watchdog Observer which watches for file changes."""
    event_handler = IndexingEventHandler(
        vector_store,
        record_manager,
        loader_cls=LogseqMarkdownLoader,
        splitter=splitter,
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

# TODO: make this configurable.
# TODO: consider summarizing the chat history when over the message limit.
CHAT_HISTORY_MESSAGES_LIMIT = 10
assert (
    CHAT_HISTORY_MESSAGES_LIMIT % 2 == 0
), "Langchain's Anthropic support breaks down if given an odd number of messages"


def interactive_loop(vector_store: VectorStore) -> None:
    """Runs the interactive REPL loop until the user exits."""
    llm = get_llm()
    retriever = vector_store.as_retriever()
    chat_history_store = InMemoryChatMessageHistory()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    rag_qa_chain = prompt | llm | StrOutputParser()

    click.echo("Enter 'exit', 'quit' or ctrl-d to quit.")
    while True:
        query = click.prompt("", prompt_suffix="> ")
        if not query:
            continue
        if query.strip().lower() in ("exit", "quit"):
            break

        rag_context = retriever.invoke(query)
        chat_history = chat_history_store.messages[:CHAT_HISTORY_MESSAGES_LIMIT]
        resp_iter = rag_qa_chain.stream(
            {
                "context": _format_docs(rag_context),
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
@click.option("--verbose", is_flag=True, default=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def main(data_dir: str, verbose: bool, debug: bool) -> None:
    load_dotenv()
    set_verbosity(verbose, debug)
    vector_store = get_vector_store()
    record_manager = get_record_manager(vector_store)
    splitter = LogseqMarkdownSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE_CHARS, chunk_overlap=0
    )
    observer = schedule_observer(
        vector_store,
        record_manager,
        splitter,
        data_dir,
    )
    try:
        docs = load_dir(data_dir)
        print(f"Loaded {len(docs)} docs")
        chunks = split_docs(splitter, docs)
        print(f"Split into {len(chunks)} chunks")
        index_result = index(
            chunks,
            record_manager,
            vector_store,
            cleanup="full",
            source_id_key="source",
        )
        print(f"Initial indexing results: {index_result}")

        # enter the main repl loop and block until the user exits
        interactive_loop(vector_store)
    except (KeyboardInterrupt, click.exceptions.Abort):
        print("Exiting...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
