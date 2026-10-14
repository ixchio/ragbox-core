# RAGBox Core

> **Turn any folder into a live, citation-ready knowledge engine for Claude Code and Cursor.**

Local hybrid retrieval, a code-aware symbol graph, exact line citations, and
automatic re-indexing—without LangChain, PyTorch, or a separate search service.

## 15-Second Quickstart

```bash
# Install RAGBox Core (CPU-only ONNX + SQLite, ~650MB, no PyTorch/CUDA)
pip install ragbox-core

# Generate .mcp.json for Claude Code / Cursor in your project root
ragbox-mcp --dir . --write-mcp-json

# Or register directly with Claude Code CLI
claude mcp add --transport stdio ragbox -- ragbox-mcp --dir .
```

Then ask your coding agent: **“Where is authentication enforced, and which
functions call it?”** RAGBox returns revision-safe citations such as
`src/auth.py:41-78`, backed by dense search, SQLite FTS5, and Personalized
PageRank over the repository symbol graph.

[![PyPI version](https://badge.fury.io/py/ragbox-core.svg)](https://badge.fury.io/py/ragbox-core)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ixchio/ragbox-core/actions/workflows/ci.yml/badge.svg)](https://github.com/ixchio/ragbox-core/actions/workflows/ci.yml)

> **RAG that works in 3 lines. Not 50.**

⭐ **If this project helps you, consider giving it a star.**

### Why developers use it

- **Codebase-native:** Python AST symbols, calls, inheritance, and multi-hop graph retrieval.
- **Agent-ready:** one-command MCP setup for Claude Code and Cursor.
- **Auditable:** every retrieved chunk carries its source revision and exact line span.
- **Local retrieval:** CPU-only ONNX embeddings, SQLite FTS5, and C-backed `igraph` PPR.
- **Always fresh:** filesystem watching, content-addressed revisions, and crash-safe restoration.

---

## The Problem With LangChain

This is a LangChain RAG app:

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

loader = DirectoryLoader("./docs", glob="**/*.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma")
vectorstore.persist()

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer from context:\n{context}\n\nQuestion: {question}"
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

result = chain({"query": "What is the vacation policy?"})
print(result["result"])
```

**That's 30+ lines. And it doesn't include GraphRAG, reranking, or streaming.**

---

## This is RAGBox

```python
from ragbox import RAGBox

rag = RAGBox("./docs")
print(rag.query("What is the vacation policy?"))
```

**3 lines. Includes dense+sparse+graph retrieval, reranking, streaming, and self-healing.**

---

## Install

```bash
pip install ragbox-core
```

Local embeddings and reranking run on CPU through FastEmbed + ONNX Runtime.
The default install does not require PyTorch, CUDA, or NVIDIA packages.

Set one API key:

```bash
export GROQ_API_KEY="gsk_..."   # free tier works
# OR
export OPENAI_API_KEY="sk-..."
# OR
export ANTHROPIC_API_KEY="sk-ant-..."
```

That's it. No config files. No boilerplate. Point it at a folder and ask questions.

### MCP server for coding agents

Expose an always-fresh local index to MCP clients without installing an extra
SDK:

```bash
ragbox-mcp --dir /absolute/path/to/project
```

Client configuration:

```json
{
  "mcpServers": {
    "ragbox": {
      "command": "ragbox-mcp",
      "args": ["--dir", "/absolute/path/to/project"]
    }
  }
}
```

The stdio server exposes `ragbox_search`, `ragbox_graph_context`, and
`ragbox_status`. Search defaults to `tri_modal_fast` and flushes queued file
changes before retrieval.

---

## What You Get For Free

| Feature | LangChain | LlamaIndex | **RAGBox** |
|---|:---:|:---:|:---:|
| Works in 3 lines | ❌ | ❌ | ✅ |
| Auto-detects LLM provider | ❌ | ❌ | ✅ |
| Built-in GraphRAG (Leiden) | ❌ | ❌ | ✅ |
| Revision-safe SQLite FTS5 sparse retrieval | manual | manual | ✅ built-in |
| Three retrieval modes (fast, tri-modal fast, full) | manual | manual | ✅ built-in |
| Dependency-free local MCP server | manual | manual | ✅ built-in |
| Cross-Encoder Reranking | manual | manual | ✅ auto |
| Multi-hop Query Decomposition | manual | manual | ✅ auto |
| Streaming (`astream()`) | complex | complex | ✅ built-in |
| Self-healing watchdog | ❌ | ❌ | ✅ |
| Crash-safe graph cold restarts | manual | manual | ✅ persisted revisions |
| Cost estimation before indexing | ❌ | ❌ | ✅ |
| Recursive semantic chunking | manual | manual | ✅ auto |
| Circuit breaker (cost protection) | ❌ | ❌ | ✅ |

---

## Real Examples

### Ask questions about your company docs
```python
from ragbox import RAGBox

rag = RAGBox("./company-docs")
print(rag.query("What's the oncall escalation policy?"))
print(rag.query("Who does Maria Santos report to?"))     # cross-doc GraphRAG
print(rag.query("What was Q4 revenue and who drove it?"))  # multi-hop
```

### Stream answers token-by-token
```python
import asyncio
from ragbox import RAGBox

async def main():
    rag = RAGBox("./docs")
    async for chunk in rag.astream("Summarize all findings"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Chat with a codebase
```python
from ragbox import RAGBox

rag = RAGBox("./my-python-project")  # auto-parses .py files with AST
print(rag.query("How does the auth middleware work?"))
print(rag.query("Which functions call the database?"))  # graph traversal
```

### Estimate cost before indexing
```python
from ragbox import RAGBox

rag = RAGBox("./large-docs")
estimate = rag.estimate_cost()
print(f"Indexing will cost ~${estimate.total_cost_usd:.4f}")
# Indexing will cost ~$0.0023
```

### Docker — instant RAG server
```bash
docker run \
  -v ./docs:/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -p 8000:8000 \
  ragbox/ragbox

# Query via HTTP
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?"}'
```

---

## How It Works

RAGBox wires up 7 components automatically — you never touch them:

```
Your Documents
     │
     ▼
Document Processor       ← auto-routes PDF / TXT / PPTX / Code
     │
     ├──▶ Chunking Engine    ← recursive semantic chunking
     │
     ├──▶ Vector Store       ← ChromaDB, context-enriched embeddings
     ├──▶ Sparse Index       ← SQLite FTS5 BM25, active revisions only
     │
     └──▶ Knowledge Graph    ← persisted revisions + Leiden/Louvain communities
                                          │
Your Question ────────────────────────────▼
     │                          Agentic Orchestrator
     │                       (classifies: vector / graph / multi-hop)
     │                              │           │
     │                    ┌─────────┘           └──────────┐
     │              Simple queries          Complex queries
     │          (dense + sparse)      (decompose + graph + dense + sparse)
     │                    │                        │
     └────────────────────┴────────────────────────┘
                               Retrieval Fusion (RRF) + Cross-Encoder Reranking
                                          │
                                     Your Answer
```

Active file pointers, sparse chunks, graph contributions, and reusable community
summaries are stored in `.ragbox_state/cas_state.db`. Graph snapshots and active
pointers commit together, so a restart restores the graph without re-indexing
unchanged files; files deleted while RAGBox was offline are reconciled at startup.

---

## CLI

```bash
# Index your documents
ragbox init ./docs

# Ask a question
ragbox query "What is the vacation policy?" -d ./docs
```

---

## When RAGBox Wins

RAGBox is **not** for everyone. Here's exactly when to use it:

✅ **Use RAGBox if:**

- You want a working RAG system today, not next week
- You're tired of wiring together 10 LangChain components
- You need cross-document reasoning (GraphRAG) without a PhD
- You're building internal tools, demos, or MVPs

❌ **Don't use RAGBox if:**

- You need highly custom retrieval pipelines
- You're building a commercial RAG product with specific SLAs
- You want to control every single component manually

---

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for scope, methodology, and historical results.

### Measured local CPU retrieval ablation

Measured on October 15, 2026 using an Intel Core i5-1135G7 laptop (8 logical
CPUs, 4 ONNX threads), Python 3.12.3, FastEmbed 0.9.0, and ONNX Runtime 1.24.2.
The run indexed 12 code/incident/architecture/policy/spec documents, evaluated
18 queries across exact, semantic, and relationship categories, and repeated
each query three times. Model warm-up is excluded from query latency; indexing
took 1.49 seconds and made zero cloud LLM calls.

| Retrieval mode | Recall@3 | MRR@5 | p50 | p95 |
|---|---:|---:|---:|---:|
| Dense-only (Chroma + BGE-small) | 0.917 | 0.894 | 9.53 ms | 11.62 ms |
| Sparse-only (SQLite FTS5 BM25) | 0.889 | 0.889 | 0.62 ms | 1.17 ms |
| Fast (dense + FTS5 RRF) | **0.972** | 0.880 | **9.89 ms** | **13.60 ms** |
| Tri-modal fast (dense + FTS5 + graph RRF) | **0.972** | 0.880 | 10.98 ms | 14.23 ms |
| Full (tri-modal + rank-fused CPU cross-encoder) | **0.972** | **0.917** | 463.17 ms | 587.47 ms |

The median Full-mode rerank stage was 447.63 ms, versus 7.49 ms embedding,
6.47 ms dense search, 1.94 ms FTS5, 0.45 ms graph lookup, and 0.11 ms fusion.
Rank blending restored Full mode to 0.972 Recall@3, while Fast remained the
latency winner. These are retrieval-only fixture results, not an end-to-end
answer-quality or competitor benchmark.

Reproduce with `poetry run python benchmarks/ablation_cpu.py`. The complete
machine-readable result is in
[`.benchmarks/ablation_cpu.json`](.benchmarks/ablation_cpu.json).

**RAGBox features three retrieval modes:**

- **Fast Path (Factual):** Weighted dense + SQLite FTS5 sparse fusion; bypasses the graph and reranker.
- **Tri-Modal Fast:** Dense + sparse + graph fusion without cross-encoder latency.
- **Full Path (GraphRAG):** Dense + sparse + graph fusion followed by bounded, rank-fused CPU cross-encoder reranking.

Full mode is intended for cases where improved first-hit ordering is worth the
CPU cost. Use Fast by default and Tri-Modal Fast when graph context is required
without cross-encoder latency.

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — use it for anything.
