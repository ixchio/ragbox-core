# RAGBox Core

**Live repository context for coding agents.**

RAGBox indexes a codebase, keeps the index in sync, and gives Claude Code,
Cursor, and other MCP clients hybrid search, symbol-graph context, and exact
`file:line-line` citations. It runs as a local sidecar, with no hosted search
service to deploy or maintain.

[![PyPI](https://img.shields.io/pypi/v/ragbox-core)](https://pypi.org/project/ragbox-core/)
[![Python](https://img.shields.io/pypi/pyversions/ragbox-core)](https://pypi.org/project/ragbox-core/)
[![CI](https://github.com/ixchio/ragbox-core/actions/workflows/ci.yml/badge.svg)](https://github.com/ixchio/ragbox-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ixchio/ragbox-core)](LICENSE)
[![M8ven Score](https://m8ven.ai/badge/mcp/ixchio-ragbox-core-1yjbbk)](https://m8ven.ai/mcp/ixchio-ragbox-core-1yjbbk?s=readme)


## Install it in a project

Set one LLM provider key for answer generation and prose-document graph
extraction. Groq and Anthropic use local CPU embeddings; OpenAI uses its
embedding API:

```bash
pip install -U ragbox-core
export GROQ_API_KEY="gsk_..."  # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

For Claude Code, create a project-scoped MCP configuration:

```bash
ragbox-mcp --dir . --write-mcp-json
claude
```

Or register it through the Claude Code CLI:

```bash
claude mcp add --scope project --transport stdio ragbox -- ragbox-mcp --dir .
```

Cursor reads project MCP servers from `.cursor/mcp.json`:

```bash
mkdir -p .cursor
ragbox-mcp --dir . --print-mcp-config > .cursor/mcp.json
```

The first start downloads the local embedding model and builds the index. The
`full` retrieval mode downloads its reranking model on first use. After that,
ask the agent questions such as:

```text
Find every caller of authorize_request and cite the exact lines.
What connects the payment retry path to idempotency handling?
Show the implementation and tests that define session expiry.
```

## What the agent gets

RAGBox exposes three MCP tools:

- `ragbox_search` returns ranked passages with revision IDs and exact spans such
  as `src/auth.py:41-78`.
- `ragbox_graph_context` returns matched symbols, entities, relationships, and
  graph-community context.
- `ragbox_status` reports index readiness, active documents, graph size, and
  file-watcher health.

The retrieval path combines:

- local BGE-small embeddings through FastEmbed and ONNX Runtime;
- SQLite FTS5 lexical search over active document revisions;
- a symbol/entity graph ranked with C-backed Personalized PageRank;
- Python AST extraction for classes, functions, methods, imports, inheritance,
  and intra-file calls without an extraction LLM;
- weighted reciprocal-rank fusion and optional CPU cross-encoder reranking;
- a file watcher that updates changed files and removes deleted revisions.

Generated artifacts, dependency directories, caches, lockfiles, and matching
`.gitignore` entries are excluded from indexing.

## Retrieval modes

- `fast` — dense + FTS5 retrieval. Use it for low-latency lookups.
- `tri_modal_fast` — dense + FTS5 + graph retrieval. This is the MCP default.
- `full` — tri-modal retrieval followed by CPU cross-encoder reranking.

Example MCP result:

```json
{
  "query": "where is request authorization enforced?",
  "mode": "tri_modal_fast",
  "citations": [
    {
      "source_path": "/work/api/auth.py",
      "revision_id": "9e34...",
      "start_line": 41,
      "end_line": 78,
      "span": "/work/api/auth.py:41-78",
      "score": 0.0317,
      "text": "..."
    }
  ]
}
```

## Python API

Use the Python API when you want RAGBox to retrieve context and generate the
answer itself:

```python
from ragbox import RAGBox

ragbox = RAGBox("./docs")
try:
    answer = ragbox.query("What is the incident escalation policy?")
    print(answer)
finally:
    ragbox.close()
```

Async applications can call `await ragbox.aquery(...)` or stream with
`ragbox.astream(...)`. Do not call the synchronous methods from an active event
loop.

RAGBox selects the first configured provider in this order: OpenAI, Anthropic,
then Groq. OpenAI-compatible endpoints can be configured with
`OPENAI_BASE_URL` and `OPENAI_MODEL`. A local GGUF model is available through
the `local` extra and `LOCAL_MODEL_PATH`.

## Supported input

The base install handles Python, JavaScript, TypeScript, TSX, Go, Rust, Java,
C, C++, Markdown, text, CSV, JSON, HTML, XML, PDF, and PowerPoint files. Image
OCR is optional:

```bash
pip install "ragbox-core[ocr]"
```

Exact line citations are most useful for code and text files. PDF and slide
content is normalized during extraction, so their reported lines refer to the
normalized text rather than the original visual page coordinates.

## HTTP server

Install the server extra and point it at a document directory:

```bash
pip install "ragbox-core[server]"
export RAGBOX_DOCUMENT_DIR=/absolute/path/to/docs
uvicorn ragbox.server:app --host 127.0.0.1 --port 8000
```

The server provides `/health`, `/query`, `/query/stream`, and `/stats`.

## How state stays correct

RAGBox stores active file revisions, sparse chunks, graph snapshots, and graph
community summaries in `.ragbox_state/cas_state.db`. A changed file is indexed
as a new content-addressed revision before the active pointer advances. Deleted
files are reconciled at startup, and stale revisions are filtered from search.

```text
files -> parse -> chunk -> dense index ----\
                    |        FTS5 index ----+-> rank fusion -> citations
                    +-> symbol/entity graph-/        |
                                                   optional reranker
```

## Measured retrieval result

The checked-in CPU ablation uses 12 fixture documents and 18 exact, semantic,
and relationship queries on an Intel i5-1135G7. In that run, `fast` retrieval
reached `0.972 Recall@3` at `16.57 ms p50`; `full` kept the same Recall@3,
improved `MRR@5` from `0.880` to `0.889`, and increased p50 latency to
`671.62 ms` because of reranking.

These are local retrieval measurements, not an end-to-end answer-quality or
competitor benchmark. See [BENCHMARKS.md](BENCHMARKS.md) for the environment,
method, raw output, and reproduction command.

## Current boundaries

- An LLM provider key or a configured local GGUF model is required for generated
  answers and prose-document entity extraction.
- Python gets AST-level relationships. Other supported languages currently use
  deterministic regex symbol extraction.
- The default embedding and reranking models are downloaded on first use.
- This project does not claim that graph retrieval or reranking improves every
  query; choose the retrieval mode based on the workload.

## Development

```bash
poetry install
poetry run pytest -q
poetry run ruff check .
poetry run mypy ragbox
poetry run black --check .
poetry run python benchmarks/ablation_cpu.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

[MIT](LICENSE)
