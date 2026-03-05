#!/usr/bin/env python3
"""
RAGBox Benchmark Suite v3 — REAL GraphRAG (no mocks)

Compares:
  - RAGBox  : full pipeline (vector + Leiden GraphRAG + reranking + query routing)
  - Vanilla : plain ChromaDB vector search + LLM answer

Scoring: Sentence-Transformer cosine similarity (all-MiniLM-L6-v2)
         Higher is better. Range [0, 1].

Usage:
    GROQ_API_KEY=... python benchmarks/run_benchmark.py
"""
import os
import sys
import time
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict
import numpy as np

# ============================================================
# Test Corpus — 8 interconnected documents
# ============================================================
DOCUMENTS = {
    "company_policy.txt": """
Company Leave Policy (Effective January 2025)

All full-time employees are entitled to 20 days of Paid Time Off (PTO) per calendar year.
PTO accrues at a rate of 1.67 days per month. Unused PTO can be carried over up to 5 days.
Employees must submit leave requests at least 2 weeks in advance through the HR portal.
Sick leave is separate from PTO and provides up to 10 days per year.
Maternity leave is 16 weeks fully paid. Paternity leave is 6 weeks fully paid.
The leave policy was updated by VP of HR Sarah Chen after employee satisfaction survey results
showed that the previous 15-day policy was causing high attrition in the engineering department.
""",
    "engineering_handbook.txt": """
Engineering Handbook - Version 3.2 (Updated by CTO James Wu)

Code Review Policy:
All code changes require at least 2 approving reviews before merge.
Reviews must be completed within 48 hours of submission.
The author must address all review comments before requesting re-review.

Deployment Process:
We use a blue-green deployment strategy with automatic rollback.
All deployments to production require a passing CI/CD pipeline.
Staging deployments happen automatically on merge to the 'develop' branch.
Production deployments require manual approval from the team lead.
The deployment infrastructure is managed by the Platform team led by Maria Santos.

Incident Response:
SEV1 incidents require immediate response within 15 minutes.
The on-call engineer must acknowledge and begin investigation immediately.
Post-mortems are required for all SEV1 and SEV2 incidents within 5 business days.
SEV1 incidents are escalated to CTO James Wu if not resolved within 1 hour.
""",
    "product_overview.txt": """
Product Overview - DataSync Pro

DataSync Pro is our flagship data integration platform.
It connects over 200 data sources including Salesforce, HubSpot, PostgreSQL, and Snowflake.
The platform processes over 50 billion records per month across our customer base.
DataSync Pro was originally developed by the DataOps team under VP of Engineering Lisa Park.

Key Features:
- Real-time CDC (Change Data Capture) with sub-second latency
- Schema drift detection and automatic resolution
- Field-level encryption for PII data — compliant with SOC2 and HIPAA requirements
- Custom transformation pipelines using Python or SQL
- Built-in data quality monitoring with alerting

Pricing:
- Starter: $299/month (up to 10M records)
- Professional: $999/month (up to 100M records)
- Enterprise: Custom pricing (unlimited records)

DataSync Pro's encryption layer was built in collaboration with the Security team led by CISO Alex Rivera.
""",
    "quarterly_report.txt": """
Q4 2025 Financial Summary (Prepared by CFO David Kim)

Revenue: $48.2M (up 23% YoY)
ARR: $185M (up 31% YoY)
Net Revenue Retention: 142%
Gross Margin: 78.5%
Free Cash Flow: $8.1M

Customer Metrics:
Total customers: 2,847 (up 18% YoY)
Enterprise customers (>$100K ARR): 312 (up 42% YoY)
Average deal size: $65K (up from $52K)

Key wins this quarter:
- Landed Fortune 100 financial services firm ($2.1M ARR) — closed by Sales VP Rachel Nguyen
- Expanded existing healthcare customer by 340% ($890K ARR)
- Launched in EMEA market with 23 new logos, led by GM Europe Thomas Schmidt

Headcount: 487 employees (up from 391 in Q4 2024)
Engineering accounts for 58% of headcount (283 engineers).
""",
    "security_policy.txt": """
Information Security Policy (Authored by CISO Alex Rivera)

Data Classification:
All company data must be classified into one of four levels:
1. Public - Can be freely shared externally
2. Internal - For employees only
3. Confidential - Need-to-know basis with access controls
4. Restricted - Requires explicit approval and audit logging

Access Control:
We follow the principle of least privilege.
All system access requires multi-factor authentication (MFA).
Service accounts must be rotated every 90 days.
SSH keys must use ED25519 or RSA-4096.

Compliance:
DataSync Pro maintains SOC2 Type II and HIPAA certifications.
Annual penetration testing is conducted by external firm CyberShield Inc.
The Security team works closely with VP of Engineering Lisa Park to ensure
all product code passes security review before production deployment.
""",
    "org_chart.txt": """
Executive Team & Organization Structure

CEO: Michael Torres (reports to Board)
CTO: James Wu (reports to CEO) — oversees all engineering
CFO: David Kim (reports to CEO) — oversees finance
VP of Engineering: Lisa Park (reports to CTO) — leads product engineering and DataOps
VP of HR: Sarah Chen (reports to CEO) — leads people operations
VP of Sales: Rachel Nguyen (reports to CEO) — leads global sales
CISO: Alex Rivera (reports to CTO) — leads security and compliance
GM Europe: Thomas Schmidt (reports to VP Sales) — leads EMEA expansion
Platform Team Lead: Maria Santos (reports to VP Engineering) — leads DevOps/infra

Total headcount breakdown:
- Engineering: 283 (58%)
- Sales & Marketing: 98 (20%)
- Operations: 54 (11%)
- G&A: 52 (11%)
""",
    "board_meeting.txt": """
Board Meeting Notes - December 2025

Key Decisions:
1. Approved 2026 budget of $62M OpEx (proposed by CFO David Kim)
2. Authorized Series C fundraise of $50M led by Accel Partners
3. Approved EMEA expansion plan — Thomas Schmidt to hire 40 people in Berlin office
4. CTO James Wu presented DataSync Pro v3.0 roadmap including AI-powered schema mapping
5. CISO Alex Rivera presented security audit results — zero critical findings

Strategic Priorities for 2026:
1. Achieve $250M ARR (from current $185M)
2. Launch DataSync Pro v3.0 with AI features by Q2
3. Reach 500 enterprise customers (from current 312)
4. Open Tokyo office under new GM Japan (to be hired)

CEO Michael Torres noted that the company's 142% net revenue retention
is a key competitive advantage and instructed Sales VP Rachel Nguyen
to focus on expansion revenue from existing enterprise accounts.
""",
    "incident_postmortem.txt": """
Post-Mortem: SEV1 Incident PM-2025-047
Date: November 15, 2025
Duration: 4 hours 23 minutes
Impact: DataSync Pro CDC pipeline down for 847 customers

Root Cause:
A schema migration deployed by the Platform team (led by Maria Santos) triggered
an unexpected cascade in the CDC engine. The migration was approved by VP of Engineering
Lisa Park but the rollback procedure failed because the blue-green deployment
(as specified in the Engineering Handbook) had a configuration drift.

Timeline:
- 02:14 UTC: Alert fired, on-call engineer (Platform team) paged
- 02:29 UTC: On-call acknowledged (15 min SLA met per Engineering Handbook)
- 02:45 UTC: Escalated to Maria Santos
- 04:00 UTC: Escalated to CTO James Wu (90 min delay beyond 1-hour policy)
- 06:37 UTC: Service restored via manual rollback

Action Items:
1. Fix blue-green deployment config drift — Owner: Maria Santos, Due: Dec 1
2. Add CDC migration dry-run step — Owner: Lisa Park, Due: Dec 15
3. Update incident escalation policy — Owner: James Wu, Due: Dec 1
4. Review with Security team — Owner: Alex Rivera, Due: Dec 15

Cost Impact: Estimated $340K in SLA credits (CFO David Kim to approve)
""",
}

# ============================================================
# QA Pairs — factual, relationship, multi-hop
# ============================================================
QA_PAIRS: List[Dict[str, str]] = [
    # --- Simple factual ---
    {
        "question": "How many days of PTO do employees get?",
        "ground_truth": "20 days of PTO per year",
        "type": "factual",
    },
    {
        "question": "What is the maternity leave policy?",
        "ground_truth": "Maternity leave is 16 weeks fully paid",
        "type": "factual",
    },
    {
        "question": "How many code review approvals are needed?",
        "ground_truth": "At least 2 approving reviews before merge",
        "type": "factual",
    },
    {
        "question": "What was Q4 2025 revenue?",
        "ground_truth": "$48.2 million, up 23% year over year",
        "type": "factual",
    },
    {
        "question": "How many data sources does DataSync Pro connect to?",
        "ground_truth": "Over 200 data sources",
        "type": "factual",
    },
    # --- Cross-document relationship (GraphRAG advantage) ---
    {
        "question": "Who does Maria Santos report to?",
        "ground_truth": "Maria Santos reports to VP of Engineering Lisa Park",
        "type": "relationship",
    },
    {
        "question": "Which executive is responsible for both security policy and DataSync Pro encryption?",
        "ground_truth": "CISO Alex Rivera leads security and built the encryption layer for DataSync Pro",
        "type": "relationship",
    },
    {
        "question": "Who updated the leave policy and why?",
        "ground_truth": "VP of HR Sarah Chen updated it because the previous 15-day policy caused high attrition in engineering",
        "type": "relationship",
    },
    {
        "question": "What went wrong in the November SEV1 incident and who was responsible?",
        "ground_truth": "A schema migration by Maria Santos' Platform team caused the CDC outage. Lisa Park approved it. CTO James Wu was escalated late.",
        "type": "relationship",
    },
    {
        "question": "How does the security team collaborate with engineering?",
        "ground_truth": "CISO Alex Rivera's security team works with VP Engineering Lisa Park on code review. They also built DataSync Pro's encryption layer together.",
        "type": "relationship",
    },
    # --- Multi-hop (3+ docs) ---
    {
        "question": "What is the company's plan to grow from $185M ARR to $250M and who is driving it?",
        "ground_truth": "CEO Michael Torres set the $250M target. Sales VP Rachel Nguyen is to focus on expansion revenue. Thomas Schmidt is expanding in EMEA. CTO James Wu is launching v3.0 with AI.",
        "type": "multi_hop",
    },
    {
        "question": "What is the relationship between the deployment strategy mentioned in the engineering handbook and the SEV1 incident?",
        "ground_truth": "The engineering handbook specifies blue-green deployment with automatic rollback, but in the SEV1 incident the rollback failed due to configuration drift in the blue-green setup.",
        "type": "multi_hop",
    },
    {
        "question": "How many engineers does the company have and what percentage of total headcount is that?",
        "ground_truth": "283 engineers, which is 58% of the 487 total employees",
        "type": "multi_hop",
    },
    {
        "question": "Who manages the infrastructure that caused the outage and who do they report to?",
        "ground_truth": "Maria Santos leads the Platform team that caused the outage. She reports to VP of Engineering Lisa Park, who reports to CTO James Wu.",
        "type": "multi_hop",
    },
    {
        "question": "What compliance certifications does the company have and who is responsible for them?",
        "ground_truth": "SOC2 Type II and HIPAA certifications, managed by CISO Alex Rivera, with annual pen testing by CyberShield Inc.",
        "type": "multi_hop",
    },
]

# Groq free tier: ~30 req/min. Sleep between LLM calls to stay safe.
INTER_CALL_SLEEP = 2.5  # seconds


def setup_test_corpus(base_dir: Path) -> Path:
    doc_dir = base_dir / "test_corpus"
    doc_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in DOCUMENTS.items():
        (doc_dir / filename).write_text(content.strip())
    return doc_dir


class SemanticScorer:
    """Sentence-Transformer cosine similarity scorer."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, answer: str, ground_truth: str) -> float:
        if not answer or not ground_truth:
            return 0.0
        embeddings = self.model.encode([answer, ground_truth])
        a, b = embeddings[0], embeddings[1]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return max(0.0, sim)


# ─── RESET ChromaDB singleton between runs ────────────────────────────────────
def _reset_chroma():
    try:
        import chromadb.api.client as cc

        cc.SharedSystemClient._identifer_to_system = {}
    except Exception:
        pass


# ─── VANILLA benchmark ────────────────────────────────────────────────────────
async def run_vanilla_benchmark(doc_dir: Path, scorer: SemanticScorer) -> List[Dict]:
    """Plain ChromaDB vector search + LLM — no graph, no reranking."""
    from ragbox.utils.embeddings import EmbeddingAutoDetector
    from ragbox.utils.vector_stores import ChromaStore
    from ragbox.utils.llm_clients import LLMAutoDetector
    from ragbox.config.defaults import Settings
    from ragbox.core.chunking_engine import FixedChunker
    from ragbox.models.documents import Document, DocumentType

    _reset_chroma()
    settings = Settings()
    embeddings = EmbeddingAutoDetector.detect(settings)
    llm = LLMAutoDetector.detect(settings)
    store = ChromaStore(persist_dir=str(doc_dir / ".vanilla_chroma"))
    chunker = FixedChunker(chunk_size=1000, overlap=200)

    print("  Indexing corpus for Vanilla…")
    for filename, content in DOCUMENTS.items():
        doc = Document(
            id=filename,
            path=doc_dir / filename,
            content=content.strip(),
            doc_type=DocumentType.TEXT,
        )
        chunks = chunker.chunk(doc)
        texts = [c.content for c in chunks]
        emb_list = await embeddings.embed_documents(texts)
        docs_to_add = [
            {
                "id": c.id,
                "content": c.content,
                "embedding": emb,
                "metadata": {"doc_id": c.document_id},
            }
            for c, emb in zip(chunks, emb_list)
        ]
        if docs_to_add:
            await store.add_documents(docs_to_add)

    results = []
    for i, qa in enumerate(QA_PAIRS):
        print(f"  [{i+1}/{len(QA_PAIRS)}] Vanilla: {qa['question'][:60]}…")
        start = time.time()
        try:
            query_emb = await embeddings.embed_query(qa["question"])
            hits = await store.search(query_emb, k=5)
            context = "\n\n".join([r.content for r in hits])
            prompt = (
                f"Answer based ONLY on this context:\n\n{context}"
                f"\n\nQuestion: {qa['question']}"
            )
            answer = await llm.agenerate(prompt, system="Be concise and accurate.")
            elapsed = (time.time() - start) * 1000
            score = scorer.score(answer, qa["ground_truth"])
            results.append(
                {
                    "question": qa["question"],
                    "answer": answer[:300],
                    "ground_truth": qa["ground_truth"],
                    "score": round(score, 3),
                    "latency_ms": round(elapsed, 0),
                    "type": qa["type"],
                    "system": "Vanilla Vector",
                }
            )
            print(f"     score={score:.3f}  latency={elapsed:.0f}ms")
        except Exception as e:
            print(f"     ERROR: {e}")
            results.append(
                {
                    "question": qa["question"],
                    "answer": f"ERROR: {str(e)[:100]}",
                    "ground_truth": qa["ground_truth"],
                    "score": 0.0,
                    "latency_ms": 0,
                    "type": qa["type"],
                    "system": "Vanilla Vector",
                }
            )
        await asyncio.sleep(INTER_CALL_SLEEP)
    return results


# ─── RAGBOX benchmark ─────────────────────────────────────────────────────────
async def run_ragbox_benchmark(doc_dir: Path, scorer: SemanticScorer) -> List[Dict]:
    """RAGBox full pipeline — real GraphRAG, real reranking, real query routing."""
    _reset_chroma()
    os.environ["CHROMA_DB_DIR"] = str(doc_dir / ".ragbox_chroma")

    from ragbox import RAGBox

    print("  Initialising RAGBox (this builds the index + Knowledge Graph)…")
    rag = RAGBox(doc_dir)

    # Wait for the async index build to finish (self_healer.initial_build)
    print("  Waiting for index to be ready…", end="", flush=True)
    for i in range(60):
        await asyncio.sleep(2)
        try:
            test_emb = await rag.embedding_provider.embed_query("test")
            hits = await rag.vector_store.search(test_emb, k=1, min_score=-1.0)
            if hits:
                print(f" ready after ~{(i+1)*2}s")
                break
        except Exception:
            print(".", end="", flush=True)
            continue
    else:
        print(" timeout — proceeding anyway")

    # Extra pause to let graph extraction finish (it runs concurrently)
    print("  Waiting 5s for Knowledge Graph construction to settle…")
    await asyncio.sleep(5)

    results = []
    for i, qa in enumerate(QA_PAIRS):
        print(f"  [{i+1}/{len(QA_PAIRS)}] RAGBox:  {qa['question'][:60]}…")
        start = time.time()
        try:
            answer = await rag.aquery(qa["question"])
            elapsed = (time.time() - start) * 1000
            score = scorer.score(answer, qa["ground_truth"])
            results.append(
                {
                    "question": qa["question"],
                    "answer": answer[:300],
                    "ground_truth": qa["ground_truth"],
                    "score": round(score, 3),
                    "latency_ms": round(elapsed, 0),
                    "type": qa["type"],
                    "system": "RAGBox",
                }
            )
            print(f"     score={score:.3f}  latency={elapsed:.0f}ms")
        except Exception as e:
            print(f"     ERROR: {e}")
            results.append(
                {
                    "question": qa["question"],
                    "answer": f"ERROR: {str(e)[:100]}",
                    "ground_truth": qa["ground_truth"],
                    "score": 0.0,
                    "latency_ms": 0,
                    "type": qa["type"],
                    "system": "RAGBox",
                }
            )
        await asyncio.sleep(INTER_CALL_SLEEP)
    return results


# ─── REPORT ───────────────────────────────────────────────────────────────────
def generate_report(ragbox_results: List[Dict], vanilla_results: List[Dict]) -> str:
    def avg_by_type(results, qtype):
        typed = [r for r in results if r["type"] == qtype]
        return sum(r["score"] for r in typed) / max(len(typed), 1)

    def avg_latency(results):
        valid = [r for r in results if r["latency_ms"] > 0]
        return sum(r["latency_ms"] for r in valid) / max(len(valid), 1)

    r_overall = sum(r["score"] for r in ragbox_results) / len(ragbox_results)
    v_overall = sum(r["score"] for r in vanilla_results) / len(vanilla_results)
    r_factual = avg_by_type(ragbox_results, "factual")
    v_factual = avg_by_type(vanilla_results, "factual")
    r_rel = avg_by_type(ragbox_results, "relationship")
    v_rel = avg_by_type(vanilla_results, "relationship")
    r_mh = avg_by_type(ragbox_results, "multi_hop")
    v_mh = avg_by_type(vanilla_results, "multi_hop")

    def winner(a, b):
        if a > b + 0.02:
            return "**RAGBox** ✅"
        elif b > a + 0.02:
            return "Vanilla"
        return "Tie"

    def lat_winner(a, b):
        return "**RAGBox** ✅" if avg_latency(a) < avg_latency(b) else "Vanilla"

    report = f"""# RAGBox Benchmark Results

> Scored using sentence-transformer cosine similarity (not word overlap).
> Higher = better. Max score: 1.0.
> Generated by real LLM calls — no mocks.

## Summary

| Category | RAGBox | Vanilla | Winner |
|---|---|---|---|
| **Overall** | {r_overall:.3f} | {v_overall:.3f} | {winner(r_overall, v_overall)} |
| **Factual** (simple lookup) | {r_factual:.3f} | {v_factual:.3f} | {winner(r_factual, v_factual)} |
| **Relationship** (cross-doc) | {r_rel:.3f} | {v_rel:.3f} | {winner(r_rel, v_rel)} |
| **Multi-Hop** (3+ docs) | {r_mh:.3f} | {v_mh:.3f} | {winner(r_mh, v_mh)} |
| **Avg Latency** | {avg_latency(ragbox_results):.0f}ms | {avg_latency(vanilla_results):.0f}ms | {lat_winner(ragbox_results, vanilla_results)} |

RAGBox: Vector + Knowledge Graph (Leiden) + Cross-Encoder Reranking + Agentic Query Routing  
Vanilla: ChromaDB vector search only

## Detailed Results

### Factual Questions (Simple Lookup)

| Question | RAGBox | Vanilla |
|---|---|---|
"""
    for rr, vr in zip(ragbox_results, vanilla_results):
        if rr["type"] == "factual":
            report += (
                f"| {rr['question'][:60]} | {rr['score']:.3f} | {vr['score']:.3f} |\n"
            )

    report += "\n### Relationship Questions (Cross-Document)\n\n| Question | RAGBox | Vanilla |\n|---|---|---|\n"
    for rr, vr in zip(ragbox_results, vanilla_results):
        if rr["type"] == "relationship":
            report += (
                f"| {rr['question'][:60]} | {rr['score']:.3f} | {vr['score']:.3f} |\n"
            )

    report += "\n### Multi-Hop Questions (3+ Documents)\n\n| Question | RAGBox | Vanilla |\n|---|---|---|\n"
    for rr, vr in zip(ragbox_results, vanilla_results):
        if rr["type"] == "multi_hop":
            report += (
                f"| {rr['question'][:60]} | {rr['score']:.3f} | {vr['score']:.3f} |\n"
            )

    report += f"""
## Sample Answers

### RAGBox — "Who does Maria Santos report to?"
> {next((r["answer"] for r in ragbox_results if "Maria Santos" in r["question"]), "N/A")}

### Vanilla — "Who does Maria Santos report to?"
> {next((r["answer"] for r in vanilla_results if "Maria Santos" in r["question"]), "N/A")}

---

## Methodology

- **Corpus**: {len(DOCUMENTS)} interconnected documents with shared entities and relationships
- **Questions**: {len(QA_PAIRS)} total — 5 factual, 5 relationship, 5 multi-hop
- **Scoring**: Sentence-Transformer (`all-MiniLM-L6-v2`) cosine similarity
- **GraphRAG**: Real LLM entity extraction — no mocks
- **Backend**: Groq (llama-3 / mixtral) — free tier

## How to Reproduce

```bash
export GROQ_API_KEY="gsk_..."   # or OPENAI_API_KEY
python benchmarks/run_benchmark.py
```
"""
    return report


# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("RAGBox Benchmark Suite v3 — Real GraphRAG")
    print("=" * 60)

    has_key = any(
        os.getenv(k) for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]
    )
    if not has_key:
        print("\nERROR: No API key found.")
        print("  Set GROQ_API_KEY=gsk_... (free tier works)")
        sys.exit(1)

    key_name = (
        "GROQ_API_KEY"
        if os.getenv("GROQ_API_KEY")
        else "OPENAI_API_KEY"
        if os.getenv("OPENAI_API_KEY")
        else "ANTHROPIC_API_KEY"
    )
    print(f"\n✅ Using API key: {key_name}")

    scorer = SemanticScorer()
    print("Semantic scorer loaded (all-MiniLM-L6-v2)\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        doc_dir = setup_test_corpus(Path(tmpdir))
        print(f"Corpus: {len(DOCUMENTS)} documents written to {doc_dir}\n")

        print("─" * 60)
        print("PHASE 1 — Vanilla Vector Search")
        print("─" * 60)
        vanilla_results = await run_vanilla_benchmark(doc_dir, scorer)
        print(f"✅ Vanilla done — {len(vanilla_results)} queries\n")

        print("─" * 60)
        print("PHASE 2 — RAGBox Full Pipeline (Real GraphRAG)")
        print("─" * 60)
        ragbox_results = await run_ragbox_benchmark(doc_dir, scorer)
        print(f"✅ RAGBox done — {len(ragbox_results)} queries\n")

        report = generate_report(ragbox_results, vanilla_results)

        report_path = Path(__file__).parent.parent / "BENCHMARKS.md"
        report_path.write_text(report)
        print(f"Report saved → {report_path}")
        print("\n" + report)


if __name__ == "__main__":
    asyncio.run(main())
