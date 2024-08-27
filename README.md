# Logseq Chat

A CLI tool for chatting with an LLM about your [Logseq](https://logseq.com/) notes.

Logseq's journal-style note taking encourages writing a lot of notes.
Sifting through them all takes time, but an LLM can help.

The LLM is prompted with the most relevant notes to your query as well as the prior chat history.
Relevant notes are retrieved using similarity search (via embeddings) and full-text search (via [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)).
The results of the two searches are combined using [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

Changed files are re-indexed in the background. This allows the tool to have access to recent changes without restarting.

This is a toy project powered by some LangChain calls and my own hackery. Still, I've found it useful.

## Example

```text
> What were the results of my H100 performance testing?

Here are some key results from the H100 testing:

1. Initial setup issues:
   - There were problems with PyTorch not being compatible with the H100 GPUs initially. [...]
   - The FasterTransformer backend library didn't initially support the sm_90 architecture required for H100s. This was fixed by modifying the CMakeList.txt file to include sm_90.

2. Performance comparisons:
   - For a 25 token test with no streaming:
     - H100: 86ms average
     - A100: 151ms average (about 75% slower than H100)
[...]
Pages:
- [[H100 Testing]]
```

## Getting Started

```bash
poetry install

echo 'OPENAI_API_KEY=sk-foo' > .env
poetry run logseq_chat --data-dir="${HOME}/logseq-data"
```

## Requirements

- Pre-existing Logseq data in markdown format
- Python 3.x (tested with 3.11)
- API key for one or multiple of: OpenAI; Anthropic; Voyage

Embeddings are provided by OpenAI or Voyage; Chat is provided by OpenAI or Anthropic.

API keys can be placed in a `.env` file in the project root or passed as environment variables (e.g. `OPENAI_API_LEY=sk-foo`).

## TODOs

- de-bounce indexing of changed files.
- include metadata (logseq properties) in embedded content.
- improve storage/caching of embeddings. generally, unified storage.
- consider using a re-ranker instead of RRF to fuse search results.
- progress bar for embedding generation.
- configurable model (beyond choosing provider).
- prompt tuning to encourage brevity but still retain details.
- handle long chat histories; store them.
- more testing
