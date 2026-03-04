# RAGBox

[![PyPI version](https://badge.fury.io/py/ragbox-core.svg)](https://badge.fury.io/py/ragbox-core)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ixchio/ragbox-core/actions/workflows/ci.yml/badge.svg)](https://github.com/ixchio/ragbox-core/actions/workflows/ci.yml)

> **RAG that works in 3 lines. Not 50.**

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

**3 lines. Includes GraphRAG, reranking, streaming, and self-healing — out of the box.**

---

## Install

```bash
pip install ragbox-core
```

Set one API key:

```bash
export GROQ_API_KEY="gsk_..."   # free tier works
# OR
export OPENAI_API_KEY="sk-..."
# OR
export ANTHROPIC_API_KEY="sk-ant-..."
```

That's it. No config files. No boilerplate. Point it at a folder and ask questions.

---

## What You Get For Free

| Feature | LangChain | LlamaIndex | **RAGBox** |
|---|:---:|:---:|:---:|
| Works in 3 lines | ❌ | ❌ | ✅ |
| Auto-detects LLM provider | ❌ | ❌ | ✅ |
| Built-in GraphRAG (Leiden) | ❌ | ❌ | ✅ |
| Cross-Encoder Reranking | manual | manual | ✅ auto |
| Streaming (`astream()`) | complex | complex | ✅ built-in |
| Self-healing watchdog | ❌ | ❌ | ✅ |
| Cost estimation before indexing | ❌ | ❌ | ✅ |
| Multi-query expansion | manual | manual | ✅ auto |
| Default install size | ~500MB | ~300MB | **~80MB** |

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
Document Processor   ← auto-routes PDF / TXT / PPTX / Code
     │
     ├──▶ Chunking Engine  ← late chunking with context awareness
     │
     ├──▶ Vector Store     ← ChromaDB, auto-configured
     │
     └──▶ Knowledge Graph  ← Leiden/Louvain entity extraction
                                          │
Your Question ────────────────────────────▼
     │                          Agentic Orchestrator
     │                       (classifies: vector / graph / multi-query)
     │                                    │
     └────────────────────────────────────▼
                               Retrieval Fusion + Reranking
                                          │
                                     Your Answer
```

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

See [BENCHMARKS.md](BENCHMARKS.md) — reproducible comparisons against vanilla vector search on multi-hop reasoning tasks.

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — use it for anything.
