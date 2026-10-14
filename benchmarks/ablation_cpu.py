"""Deterministic, retrieval-only CPU ablation for RAGBox.

The benchmark uses the production FastEmbed, Chroma, SQLite FTS5, graph, RRF,
and cross-encoder implementations. Graph extraction is pre-seeded from the
fixture manifest so the run makes no cloud LLM calls.

Usage:
    poetry run python benchmarks/ablation_cpu.py
    poetry run python benchmarks/ablation_cpu.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import igraph as ig
import numpy as np
from loguru import logger

from ragbox.core.chunking_engine import ChunkingEngine
from ragbox.core.document_processor import DocumentProcessorRouter
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.core.retrieval_fusion import RetrievalFusionEngine
from ragbox.core.self_healing import SelfHealer
from ragbox.models.documents import Document
from ragbox.models.queries import Source
from ragbox.utils.embeddings import FastEmbedProvider
from ragbox.utils.vector_stores import ChromaStore


@dataclass(frozen=True)
class CorpusDocument:
    filename: str
    evidence_id: str
    content: str
    entities: tuple[tuple[str, str], ...]
    relations: tuple[tuple[str, str, str, str], ...] = ()


@dataclass(frozen=True)
class QueryCase:
    category: str
    query: str
    relevant: frozenset[str]


@dataclass
class ModeResult:
    recall_at_3: float
    mrr_at_5: float
    p50_ms: float
    p95_ms: float
    embed_ms: float
    dense_ms: float
    fts_ms: float
    graph_ms: float
    fusion_ms: float
    rerank_ms: float


STAGE_NAMES = (
    "embed_ms",
    "dense_ms",
    "fts_ms",
    "graph_ms",
    "fusion_ms",
    "rerank_ms",
)


CORPUS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        "checkout.py",
        "checkout-code",
        """# [DOC_ID:checkout-code]
PAYMENT_RETRY_JITTER_KEY = "payments.retry_jitter_ms"

async def finalize_checkout(order, idempotency_key):
    \"\"\"Send a confirmed purchase to Orion Payments exactly once.\"\"\"
    payment = await orion.charge(order.total, idempotency_key=idempotency_key)
    await ledger.record(payment.transaction_id)
    await atlas_queue.publish("OrderCommitted.v3", order.id)
    if payment.error_code == "PAY-4097":
        raise DuplicateLedgerWrite(payment.transaction_id)
    return payment
""",
        (
            ("Mercury Checkout", "Checkout service that confirms purchases"),
            ("Orion Payments", "Payment service receiving checkout charges"),
            ("Ledger API", "Durable financial transaction record"),
            ("Atlas Queue", "Asynchronous order event queue"),
            ("finalize_checkout", "Checkout commit function"),
            ("PAY-4097", "Duplicate ledger write error"),
        ),
        (
            (
                "finalize_checkout",
                "Orion Payments",
                "CALLS",
                "finalize_checkout sends each charge with an idempotency key",
            ),
            (
                "Orion Payments",
                "Ledger API",
                "WRITES_TO",
                "Successful payment transactions are recorded in Ledger API",
            ),
            (
                "Mercury Checkout",
                "Atlas Queue",
                "PUBLISHES_TO",
                "Checkout publishes OrderCommitted.v3 events",
            ),
        ),
    ),
    CorpusDocument(
        "auth_gateway.py",
        "auth-code",
        """# [DOC_ID:auth-code]
TOKEN_SKEW_KEY = "security.token_clock_skew_seconds"

def verify_rotating_nonce(token, chronos_time):
    \"\"\"Reject replayed login tokens while tolerating bounded clock drift.\"\"\"
    if abs(token.issued_at - chronos_time) > TOKEN_SKEW_KEY:
        raise AuthenticationError("AUTH-2719")
    return nonce_store.consume_once(token.nonce)
""",
        (
            ("Aegis Gateway", "Identity service validating login tokens"),
            ("Chronos Clock", "Authoritative time synchronization service"),
            ("verify_rotating_nonce", "Replay-safe nonce validation function"),
            ("AUTH-2719", "Authentication clock-drift error"),
        ),
        (
            (
                "Aegis Gateway",
                "Chronos Clock",
                "DEPENDS_ON",
                "Aegis compares token issue time with Chronos Clock",
            ),
        ),
    ),
    CorpusDocument(
        "cache_config.json",
        "cache-config",
        """{
  "_marker": "[DOC_ID:cache-config]",
  "cluster": "redis_cluster_beta",
  "cache.eviction.grace_ms": 45000,
  "cache.write_through.enabled": true,
  "notes": "Expired entries remain readable during the grace window."
}
""",
        (
            ("Redis Cluster Beta", "Shared cache for catalog responses"),
            ("Catalog API", "Product catalog read service"),
        ),
        (
            (
                "Catalog API",
                "Redis Cluster Beta",
                "CACHES_IN",
                "Catalog responses use the shared Redis cache",
            ),
        ),
    ),
    CorpusDocument(
        "payments_incident.md",
        "payments-incident",
        """<!-- [DOC_ID:payments-incident] -->
# PAY-4097 duplicate settlement incident

Orion Payments retried a completed Ledger API write after an Atlas Queue
acknowledgement timeout. The missing idempotency check created duplicate
settlement records. Priya Nair coordinated mitigation for Payments Platform.
The permanent fix stores the transaction key before publishing the event.
""",
        (
            ("PAY-4097", "Duplicate settlement incident"),
            ("Orion Payments", "Payment authorization and settlement service"),
            ("Ledger API", "Durable settlement record system"),
            ("Atlas Queue", "Order event delivery queue"),
            ("Priya Nair", "Payments Platform engineering lead"),
            ("Payments Platform", "Team operating payment services"),
        ),
        (
            ("PAY-4097", "Orion Payments", "AFFECTED", "Duplicate settlements"),
            (
                "Priya Nair",
                "PAY-4097",
                "RESPONDED_TO",
                "Priya coordinated incident mitigation",
            ),
        ),
    ),
    CorpusDocument(
        "auth_incident.md",
        "auth-incident",
        """<!-- [DOC_ID:auth-incident] -->
# AUTH-2719 token rejection incident

A Chronos Clock drift made Aegis Gateway reject valid rotating nonces. Elena
Park led the Identity Platform response. The remediation added bounded skew
tolerance and alerts when regional clocks diverge by more than two seconds.
""",
        (
            ("AUTH-2719", "Valid login token rejection incident"),
            ("Aegis Gateway", "Identity token gateway"),
            ("Chronos Clock", "Regional clock synchronization service"),
            ("Elena Park", "Identity Platform engineering lead"),
            ("Identity Platform", "Team operating authentication services"),
        ),
        (
            ("AUTH-2719", "Aegis Gateway", "AFFECTED", "Valid tokens rejected"),
            (
                "Chronos Clock",
                "AUTH-2719",
                "CAUSED",
                "Clock drift exceeded token tolerance",
            ),
        ),
    ),
    CorpusDocument(
        "service_architecture.md",
        "service-architecture",
        """<!-- [DOC_ID:service-architecture] -->
# Commerce service architecture

Mercury Checkout calls Orion Payments to authorize purchases. Orion Payments
writes final transactions to Ledger API. Mercury publishes OrderCommitted.v3
events to Atlas Queue, where downstream fulfillment workers consume them.
Aegis Gateway depends on Chronos Clock for replay-safe token validation.
""",
        (
            ("Mercury Checkout", "Customer checkout orchestration service"),
            ("Orion Payments", "Purchase authorization service"),
            ("Ledger API", "Financial system of record"),
            ("Atlas Queue", "Order event transport"),
            ("OrderCommitted.v3", "Committed-order event schema"),
            ("Aegis Gateway", "Login token gateway"),
            ("Chronos Clock", "Time synchronization dependency"),
        ),
        (
            ("Mercury Checkout", "Orion Payments", "CALLS", "Purchase authorization"),
            ("Orion Payments", "Ledger API", "WRITES_TO", "Final transactions"),
            ("Mercury Checkout", "Atlas Queue", "PUBLISHES_TO", "Committed orders"),
            ("Atlas Queue", "OrderCommitted.v3", "CARRIES", "Order event schema"),
            ("Aegis Gateway", "Chronos Clock", "DEPENDS_ON", "Token timestamps"),
        ),
    ),
    CorpusDocument(
        "service_ownership.md",
        "service-ownership",
        """<!-- [DOC_ID:service-ownership] -->
# Service ownership registry

Priya Nair leads Payments Platform, the team that owns Orion Payments and
Ledger API. Marco Ruiz leads Messaging Infrastructure, which owns Atlas Queue.
Elena Park leads Identity Platform, which operates Aegis Gateway. Chronos Clock
is operated by Core Reliability under Sam Chen.
""",
        (
            ("Priya Nair", "Payments Platform engineering lead"),
            ("Payments Platform", "Owner of payment systems"),
            ("Orion Payments", "Payment service"),
            ("Ledger API", "Financial record service"),
            ("Marco Ruiz", "Messaging Infrastructure engineering lead"),
            ("Messaging Infrastructure", "Owner of asynchronous messaging"),
            ("Atlas Queue", "Order event queue"),
            ("Elena Park", "Identity Platform engineering lead"),
            ("Identity Platform", "Owner of identity systems"),
            ("Aegis Gateway", "Authentication gateway"),
            ("Sam Chen", "Core Reliability engineering lead"),
            ("Core Reliability", "Owner of shared reliability services"),
            ("Chronos Clock", "Shared time service"),
        ),
        (
            ("Priya Nair", "Payments Platform", "LEADS", "Team leadership"),
            ("Payments Platform", "Orion Payments", "OWNS", "Service ownership"),
            ("Payments Platform", "Ledger API", "OWNS", "Service ownership"),
            ("Marco Ruiz", "Messaging Infrastructure", "LEADS", "Team leadership"),
            ("Messaging Infrastructure", "Atlas Queue", "OWNS", "Service ownership"),
            ("Elena Park", "Identity Platform", "LEADS", "Team leadership"),
            ("Identity Platform", "Aegis Gateway", "OWNS", "Service ownership"),
            ("Sam Chen", "Core Reliability", "LEADS", "Team leadership"),
            ("Core Reliability", "Chronos Clock", "OWNS", "Service ownership"),
        ),
    ),
    CorpusDocument(
        "refund_policy.md",
        "refund-policy",
        """<!-- [DOC_ID:refund-policy] -->
# Post-settlement reimbursement policy

Customer reimbursements above USD 500 require a human risk analyst to approve
the request. Smaller reimbursements may be automatically accepted after order
and fraud checks. Completed bank settlement cannot be reversed automatically;
the support workflow creates a compensating payment instead.
""",
        (("Risk Operations", "Human review team for large reimbursements"),),
    ),
    CorpusDocument(
        "retention_policy.md",
        "retention-policy",
        """<!-- [DOC_ID:retention-policy] -->
# Lifecycle and privacy controls

Inactive customer profiles are purged by the nightly privacy sweeper after
thirteen months. Legal-hold records are excluded. The erasure job removes the
profile from primary storage and emits a tombstone for downstream replicas.
""",
        (("Privacy Sweeper", "Nightly inactive-profile erasure worker"),),
    ),
    CorpusDocument(
        "semantic_search.py",
        "semantic-search-code",
        """# [DOC_ID:semantic-search-code]
class SemanticEncoder:
    \"\"\"Map natural-language intent into dense vector coordinates.\"\"\"

    def embed_query(self, question: str) -> list[float]:
        return self.model.query_embed(question)

    def nearest_context(self, question: str):
        vector = self.embed_query(question)
        return self.index.cosine_neighbors(vector)
""",
        (("Semantic Encoder", "Maps language meaning to dense vector coordinates"),),
    ),
    CorpusDocument(
        "event_spec.md",
        "event-spec",
        """<!-- [DOC_ID:event-spec] -->
# OrderCommitted.v3

Schema ID: evt_order_committed_v3. Required fields are order_id, customer_id,
currency, gross_amount, payment_transaction_id, and committed_at. Producers
must preserve the event_id for idempotent consumers.
""",
        (
            ("OrderCommitted.v3", "Committed-order event schema"),
            ("Atlas Queue", "Transport for committed-order events"),
        ),
        (("Atlas Queue", "OrderCommitted.v3", "CARRIES", "Versioned order event"),),
    ),
    CorpusDocument(
        "degraded_checkout_runbook.md",
        "checkout-runbook",
        """<!-- [DOC_ID:checkout-runbook] -->
# Checkout degradation runbook

When Orion Payments exceeds 800 ms p95 latency, Mercury Checkout enters safe
degraded mode, pauses automatic retries, and pages Payments Platform. Operators
must inspect Ledger API for a committed transaction before replaying a charge.
""",
        (
            ("Mercury Checkout", "Checkout orchestration service"),
            ("Orion Payments", "Payment authorization service"),
            ("Ledger API", "Financial transaction system of record"),
            ("Payments Platform", "Payment service owner"),
        ),
        (
            ("Mercury Checkout", "Orion Payments", "CALLS", "Checkout payment"),
            ("Payments Platform", "Orion Payments", "OPERATES", "On-call response"),
        ),
    ),
)


QUERIES: tuple[QueryCase, ...] = (
    QueryCase("exact", "PAY-4097", frozenset({"payments-incident"})),
    QueryCase("exact", "verify_rotating_nonce", frozenset({"auth-code"})),
    QueryCase("exact", "cache.eviction.grace_ms", frozenset({"cache-config"})),
    QueryCase("exact", "AUTH-2719", frozenset({"auth-incident"})),
    QueryCase("exact", "OrderCommitted.v3 required fields", frozenset({"event-spec"})),
    QueryCase("exact", "payments.retry_jitter_ms", frozenset({"checkout-code"})),
    QueryCase(
        "semantic",
        "Which component turns a shopper's purchase into a durable financial record?",
        frozenset({"service-architecture"}),
    ),
    QueryCase(
        "semantic",
        "How does the platform avoid performing the same money operation twice?",
        frozenset({"payments-incident", "checkout-code"}),
    ),
    QueryCase(
        "semantic",
        "What protects users when regional computer clocks disagree during login?",
        frozenset({"auth-incident", "auth-code"}),
    ),
    QueryCase(
        "semantic",
        "Where are dormant customer records removed automatically?",
        frozenset({"retention-policy"}),
    ),
    QueryCase(
        "semantic",
        "What converts natural-language meaning into coordinates for similarity?",
        frozenset({"semantic-search-code"}),
    ),
    QueryCase(
        "semantic",
        "Who reviews unusually large reimbursements?",
        frozenset({"refund-policy"}),
    ),
    QueryCase(
        "relationship",
        "How is Priya Nair connected to Ledger API?",
        frozenset({"service-ownership", "service-architecture"}),
    ),
    QueryCase(
        "relationship",
        "Which queue receives events from the service owned by Priya Nair?",
        frozenset({"service-ownership", "service-architecture"}),
    ),
    QueryCase(
        "relationship",
        "What service operated by Elena Park depends on Chronos Clock?",
        frozenset({"service-ownership", "service-architecture"}),
    ),
    QueryCase(
        "relationship",
        "Who owns Atlas Queue and what service publishes to it?",
        frozenset({"service-ownership", "service-architecture"}),
    ),
    QueryCase(
        "relationship",
        "How does Mercury Checkout reach Ledger API?",
        frozenset({"service-architecture", "checkout-code"}),
    ),
    QueryCase(
        "relationship",
        "Which team responds when PAY-4097 affects Orion Payments?",
        frozenset({"payments-incident", "service-ownership"}),
    ),
)


class SeededKnowledgeGraph(OptimizedKnowledgeGraph):
    """Use fixture-owned graph facts while exercising the production graph."""

    def __init__(self, corpus: Sequence[CorpusDocument]) -> None:
        super().__init__(llm_client=None)
        self._facts = {document.filename: document for document in corpus}

    async def prepare_document(
        self, document: Document
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        fixture = self._facts[document.path.name]
        entities = [
            {"name": name, "description": description}
            for name, description in fixture.entities
        ]
        relationships = [
            {
                "source": source,
                "target": target,
                "type": relation_type,
                "context": context,
            }
            for source, target, relation_type, context in fixture.relations
        ]
        return entities, relationships


class ForcedFullRetrievalFusionEngine(RetrievalFusionEngine):
    """Disable the production early-exit only to isolate reranker contribution."""

    def _should_skip_rerank(
        self,
        vector_results: list[Any],
        graph_results: Any,
        sparse_results: tuple[Any, ...] | list[Any] = (),
    ) -> bool:
        return False


class ProfiledRetrievalFusionEngine(ForcedFullRetrievalFusionEngine):
    """Measure production retrieval stages without changing their behavior."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_stage_timings = self._empty_stage_timings()

    @staticmethod
    def _empty_stage_timings() -> dict[str, float]:
        return {stage: 0.0 for stage in STAGE_NAMES}

    def reset_stage_timings(self) -> None:
        self.last_stage_timings = self._empty_stage_timings()

    def _record(self, stage: str, started: float) -> None:
        self.last_stage_timings[stage] += (time.perf_counter() - started) * 1000.0

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fast_mode: bool = False,
        mode: str | None = None,
    ) -> list[Source]:
        self.reset_stage_timings()
        return await super().retrieve(query, top_k, fast_mode, mode)

    async def _embed_query(self, query: str) -> list:
        started = time.perf_counter()
        result = await super()._embed_query(query)
        self._record("embed_ms", started)
        return result

    async def _search_dense(self, query_emb: list, limit: int) -> list:
        started = time.perf_counter()
        result = await super()._search_dense(query_emb, limit)
        self._record("dense_ms", started)
        return result

    async def _search_sparse(self, query: str, limit: int) -> list:
        started = time.perf_counter()
        result = await super()._search_sparse(query, limit)
        self._record("fts_ms", started)
        return result

    async def _query_graph(self, query: str) -> Any:
        started = time.perf_counter()
        result = await super()._query_graph(query)
        self._record("graph_ms", started)
        return result

    def _rrf(
        self,
        vector_res: list,
        graph_res: Any,
        sparse_res: tuple | list = (),
        k: int = 60,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        result = super()._rrf(vector_res, graph_res, sparse_res, k)
        self._record("fusion_ms", started)
        return result

    def _blend_rank_scores(
        self, reranked: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        result = super()._blend_rank_scores(reranked)
        self._record("fusion_ms", started)
        return result

    async def _rerank_candidates(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        result = await super()._rerank_candidates(query, candidates)
        self._record("rerank_ms", started)
        return result


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.casefold().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _evidence_ids(source: Source, revisions: dict[str, str]) -> set[str]:
    evidence = set(re.findall(r"\[DOC_ID:([a-z0-9-]+)\]", source.text))
    mapped = revisions.get(source.document_id)
    if mapped:
        evidence.add(mapped)
    return evidence


def _score_query(
    sources: Sequence[Source], expected: frozenset[str], revisions: dict[str, str]
) -> tuple[float, float]:
    found: set[str] = set()
    for source in sources[:3]:
        found.update(_evidence_ids(source, revisions) & expected)
    recall = len(found) / len(expected)

    reciprocal_rank = 0.0
    for rank, source in enumerate(sources[:5], start=1):
        if _evidence_ids(source, revisions) & expected:
            reciprocal_rank = 1.0 / rank
            break
    return recall, reciprocal_rank


def _write_corpus(root: Path) -> None:
    for document in CORPUS:
        (root / document.filename).write_text(document.content, encoding="utf-8")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(0)
    ig.set_random_number_generator(random.Random(0))
    logical_cpus = os.cpu_count() or 1
    onnx_threads = args.threads or max(1, logical_cpus // 2)
    os.environ["RAGBOX_ONNX_THREADS"] = str(onnx_threads)
    selected_queries = QUERIES[:6] if args.quick else QUERIES
    repeats = 1 if args.quick else args.repeats

    logger.remove()
    with tempfile.TemporaryDirectory(prefix="ragbox-ablation-") as temp_dir:
        root = Path(temp_dir)
        corpus_dir = root / "corpus"
        corpus_dir.mkdir()
        _write_corpus(corpus_dir)

        started = time.perf_counter()
        embeddings = FastEmbedProvider("BAAI/bge-small-en-v1.5")
        vector_store = ChromaStore(str(root / "chroma"))
        graph = SeededKnowledgeGraph(CORPUS)
        chunker = ChunkingEngine(llm_client=None, embedding_provider=embeddings)  # type: ignore[arg-type]
        healer = SelfHealer(
            document_dir=corpus_dir,
            document_processor=DocumentProcessorRouter(),
            chunking_engine=chunker,
            vector_store=vector_store,
            knowledge_graph=graph,
        )
        await healer.initial_build()
        index_seconds = time.perf_counter() - started

        revisions = {
            revision: next(
                item.evidence_id for item in CORPUS if item.filename == Path(path).name
            )
            for path, revision in healer.cas.active_revisions().items()
        }
        engine = ProfiledRetrievalFusionEngine(
            vector_store=vector_store,
            knowledge_graph=graph,
            embedding_provider=embeddings,
            llm_client=None,  # type: ignore[arg-type]
            sparse_search_fn=healer.cas.search_fts,
        )
        engine.set_active_revision_resolver(healer.is_active_revision)

        async def dense_only(query: str) -> list[Source]:
            engine.reset_stage_timings()
            query_embedding = await engine._embed_query(query)
            hits = engine._active_only(await engine._search_dense(query_embedding, 5))
            return [
                Source(
                    document_id=hit.metadata.get("doc_id", "unknown"),
                    text=hit.content,
                    score=hit.score,
                )
                for hit in hits[:5]
            ]

        async def sparse_only(query: str) -> list[Source]:
            engine.reset_stage_timings()
            hits = await engine._search_sparse(query, 5)
            return [
                Source(
                    document_id=hit.get("metadata", {}).get("doc_id", "unknown"),
                    text=hit.get("content", ""),
                    score=hit.get("score", 0.0),
                )
                for hit in hits
            ]

        async def fast_mode(query: str) -> list[Source]:
            return await engine.retrieve(query, top_k=5, mode="fast")

        async def tri_modal_fast_mode(query: str) -> list[Source]:
            return await engine.retrieve(query, top_k=5, mode="tri_modal_fast")

        async def full_mode(query: str) -> list[Source]:
            return await engine.retrieve(query, top_k=5, mode="full")

        modes: tuple[tuple[str, Callable[[str], Awaitable[list[Source]]]], ...] = (
            ("Dense-Only (Chroma BGE-small)", dense_only),
            ("Sparse-Only (SQLite FTS5 BM25)", sparse_only),
            ("Fast Mode (Dense + FTS5 RRF)", fast_mode),
            ("Tri-Modal Fast (Dense + FTS5 + Graph RRF)", tri_modal_fast_mode),
            ("Full Mode (Tri-Modal + CPU Cross-Encoder)", full_mode),
        )

        # Exclude one-time ONNX model initialization from warm-query latency.
        for _, retrieve in modes:
            await retrieve(selected_queries[0].query)
        if engine.reranker._model is None:
            raise RuntimeError("Cross-encoder failed to initialize; benchmark aborted")

        results: dict[str, ModeResult] = {}
        category_recall: dict[str, dict[str, float]] = {}
        for mode_name, retrieve in modes:
            recalls: list[float] = []
            reciprocal_ranks: list[float] = []
            latencies_ms: list[float] = []
            stage_values: dict[str, list[float]] = {stage: [] for stage in STAGE_NAMES}
            category_values: dict[str, list[float]] = {}
            for _ in range(repeats):
                for case in selected_queries:
                    query_started = time.perf_counter_ns()
                    sources = await retrieve(case.query)
                    elapsed_ms = (time.perf_counter_ns() - query_started) / 1_000_000
                    recall, reciprocal_rank = _score_query(
                        sources, case.relevant, revisions
                    )
                    recalls.append(recall)
                    reciprocal_ranks.append(reciprocal_rank)
                    latencies_ms.append(elapsed_ms)
                    for stage in STAGE_NAMES:
                        stage_values[stage].append(engine.last_stage_timings[stage])
                    category_values.setdefault(case.category, []).append(recall)

            results[mode_name] = ModeResult(
                recall_at_3=float(np.mean(recalls)),
                mrr_at_5=float(np.mean(reciprocal_ranks)),
                p50_ms=_percentile(latencies_ms, 50),
                p95_ms=_percentile(latencies_ms, 95),
                **{
                    stage: _percentile(values, 50)
                    for stage, values in stage_values.items()
                },
            )
            category_recall[mode_name] = {
                category: float(np.mean(values))
                for category, values in category_values.items()
            }

    metadata = {
        "cpu": _cpu_model(),
        "logical_cpus": logical_cpus,
        "onnx_threads": onnx_threads,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "fastembed": version("fastembed"),
        "onnxruntime": version("onnxruntime"),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "documents": len(CORPUS),
        "queries": len(selected_queries),
        "repeats": repeats,
        "index_seconds": index_seconds,
        "cloud_llm_calls": 0,
    }
    return {
        "metadata": metadata,
        "results": {name: asdict(result) for name, result in results.items()},
        "category_recall_at_3": category_recall,
    }


def _markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# RAGBox deterministic CPU retrieval ablation",
        "",
        (
            f"CPU: {metadata['cpu']} | logical CPUs: {metadata['logical_cpus']} | "
            f"ONNX threads: {metadata['onnx_threads']}"
        ),
        (
            f"Corpus: {metadata['documents']} documents | queries: "
            f"{metadata['queries']} | repeats: {metadata['repeats']} | "
            f"index: {metadata['index_seconds']:.2f} s | cloud LLM calls: 0"
        ),
        "",
        "| Mode | Recall@3 | MRR@5 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["results"].items():
        lines.append(
            f"| {name} | {result['recall_at_3']:.3f} | "
            f"{result['mrr_at_5']:.3f} | {result['p50_ms']:.2f} | "
            f"{result['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Median stage latency (ms):",
            "",
            "| Mode | Embed | Dense | FTS5 | Graph | Fusion | Rerank |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in report["results"].items():
        lines.append(
            f"| {name} | {result['embed_ms']:.2f} | "
            f"{result['dense_ms']:.2f} | {result['fts_ms']:.2f} | "
            f"{result['graph_ms']:.2f} | {result['fusion_ms']:.2f} | "
            f"{result['rerank_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Category Recall@3:",
            "",
            "| Mode | Exact | Semantic | Relationship |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, categories in report["category_recall_at_3"].items():
        lines.append(
            f"| {name} | {categories.get('exact', 0.0):.3f} | "
            f"{categories.get('semantic', 0.0):.3f} | "
            f"{categories.get('relationship', 0.0):.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run six exact-identifier queries once for a smoke check.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Measured repetitions per query."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX CPU threads; defaults to half the logical CPU count.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(".benchmarks/ablation_cpu.json"),
        help="Machine-readable output path.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.threads < 0:
        parser.error("--threads cannot be negative")

    report = asyncio.run(_run(args))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(_markdown(report))
    print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()
