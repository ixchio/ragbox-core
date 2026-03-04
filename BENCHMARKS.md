# RAGBox Benchmark Results

> Scored using sentence-transformer cosine similarity (not word overlap).
> Higher = better. Max score: 1.0.

## Summary

| Category | RAGBox | Vanilla | Winner |
|---|---|---|---|
| **Overall** | 0.000 | 0.000 | Tie |
| **Factual** (simple lookup) | 0.000 | 0.000 | Tie |
| **Relationship** (cross-doc) | 0.000 | 0.000 | Tie |
| **Multi-Hop** (3+ docs) | 0.000 | 0.000 | Tie |
| **Avg Latency** | 0ms | 0ms | RAGBox |

RAGBox: Vector + Knowledge Graph (Leiden) + Cross-Encoder Reranking + Query Routing
Vanilla: ChromaDB vector search only

## Detailed Results

### Factual Questions (Simple Lookup)

| Question | RAGBox | Vanilla |
|---|---|---|
| How many days of PTO do employees get? | 0.000 | 0.000 |
| What is the maternity leave policy? | 0.000 | 0.000 |
| How many code review approvals are needed? | 0.000 | 0.000 |
| What was Q4 2025 revenue? | 0.000 | 0.000 |
| How many data sources does DataSync Pro connect to? | 0.000 | 0.000 |

### Relationship Questions (Cross-Document)

| Question | RAGBox | Vanilla |
|---|---|---|
| Who does Maria Santos report to? | 0.000 | 0.000 |
| Which executive is responsible for both security policy | 0.000 | 0.000 |
| Who updated the leave policy and why? | 0.000 | 0.000 |
| What went wrong in the November SEV1 incident and who w | 0.000 | 0.000 |
| How does the security team collaborate with engineering | 0.000 | 0.000 |

### Multi-Hop Questions (3+ Documents)

| Question | RAGBox | Vanilla |
|---|---|---|
| What is the company's plan to grow from $185M ARR to $2 | 0.000 | 0.000 |
| What is the relationship between the deployment strateg | 0.000 | 0.000 |
| How many engineers does the company have and what perce | 0.000 | 0.000 |
| Who manages the infrastructure that caused the outage a | 0.000 | 0.000 |
| What compliance certifications does the company have an | 0.000 | 0.000 |

## Methodology

- **Corpus**: 8 interconnected documents with shared entities and relationships
- **Questions**: 15 total — 5 factual, 5 relationship, 5 multi-hop
- **Scoring**: Sentence-Transformer (`all-MiniLM-L6-v2`) cosine similarity between generated answer and ground truth
- **Reproducibility**: All documents embedded in the script. Run `python benchmarks/run_benchmark.py`

## How to Reproduce

```bash
export GROQ_API_KEY="gsk_..."  # or OPENAI_API_KEY
python benchmarks/run_benchmark.py
```
