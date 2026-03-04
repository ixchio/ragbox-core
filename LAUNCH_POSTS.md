# RAGBox — Launch Posts

Ready-to-deploy posts for Reddit, Hacker News, and Twitter/X.

---

## Reddit — r/LocalLLaMA / r/LangChain / r/MachineLearning

**Title:**
> I built a RAG framework that does everything LangChain does in 3 lines instead of 50

**Body:**

```
I've been using LangChain for RAG for 2 years. I got tired of wiring together 
DirectoryLoader → TextSplitter → Embeddings → VectorStore → Chain → Prompt → Retriever 
every single project.

So I built RAGBox. Here's the same app in 3 lines:

    from ragbox import RAGBox
    rag = RAGBox("./docs")
    print(rag.query("What is the vacation policy?"))

LangChain equivalent: 31 lines.
LlamaIndex equivalent: 18 lines.

What you get automatically (no config):
✅ GraphRAG (Leiden/Louvain entity extraction)
✅ Cross-encoder reranking
✅ Streaming (`astream()`)
✅ Self-healing watchdog (file changes auto-update the index)
✅ Cost estimation before indexing
✅ Works with OpenAI / Anthropic / Groq (auto-detected)

The part that surprised me: GraphRAG is where it really shines. 
Questions like "Who does Maria report to?" or "How did the outage affect Q4?" 
automatically get routed through the knowledge graph instead of vector search.

No ontology setup. No Neo4j. Just $pip install ragbox-core$.

GitHub: https://github.com/ixchio/ragbox-core
PyPI: https://pypi.org/project/ragbox-core/
Comparison: [LangChain vs RAGBox side-by-side](https://github.com/ixchio/ragbox-core/blob/master/COMPARISON.md)

Honest caveat: if you need highly custom retrieval pipelines, use LangChain. 
RAGBox is for people who want to solve their problem today, not build infrastructure.

Happy to answer questions!
```

---

## Hacker News — Show HN

**Title:**
> Show HN: RAGBox – GraphRAG in 3 lines, no LangChain required

**Text:**

```
RAGBox is a Python RAG framework where `from ragbox import RAGBox; rag = RAGBox("./docs"); rag.query("...")` 
is a complete, production-usable RAG system with GraphRAG, reranking, and streaming.

The motivation: LangChain needs ~31 lines to do basic Q&A. 
RAGBox needs 3. Both connect your documents to an LLM and answer questions.

What's different:
- Auto-GraphRAG using Leiden community detection (no ontology setup)
- Auto-routing: simple questions go to vector search, complex ones go to the knowledge graph
- Cost estimation before indexing runs
- Self-healing: watchdog detects file changes and updates the index

Honest weakness: the graph extraction uses LLM calls which costs tokens.
On Groq's free tier, it processes 8 documents before hitting rate limits.

GitHub: https://github.com/ixchio/ragbox-core
```

---

## Twitter/X Thread

**Tweet 1 (hook):**
```
LangChain RAG app: 31 lines.
LlamaIndex RAG app: 18 lines.
RAGBox RAG app: 3 lines.

Same result. Auto-GraphRAG included.

Here's how: 🧵
```

**Tweet 2:**
```
from ragbox import RAGBox

rag = RAGBox("./docs")
print(rag.query("What is the vacation policy?"))

That's it. Auto-detects OpenAI/Anthropic/Groq.
Auto-builds a Knowledge Graph.
Auto-configures ChromaDB.
Auto-reranks with Cross-Encoder.
```

**Tweet 3:**
```
The magic part: query routing.

Simple question → vector search (fast)
"Who does Maria report to?" → Knowledge Graph (cross-doc)
"How did the outage affect Q4?" → Multi-hop reasoning (3+ docs)

Zero config. You just ask questions.
```

**Tweet 4:**
```
Streaming works too:

async for chunk in rag.astream("Summarize findings"):
    print(chunk, end="")

Works with GPT-4o, Claude Sonnet, Llama 3.3 on Groq.
```

**Tweet 5 (CTA):**
```
pip install ragbox-core

Compare it yourself vs LangChain:
github.com/ixchio/ragbox-core/blob/master/COMPARISON.md

If you're tired of wiring together 10 components, this is for you.
```

---

## Reply to "LangChain is too complex" tweets

```
Built this for you → RAGBox. 3 lines for a complete RAG app with built-in GraphRAG.
pip install ragbox-core | github.com/ixchio/ragbox-core
```
