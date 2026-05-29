"""
Layer 5: RETRIEVAL FUSION
Hybrid Dense (vector) + Graph (knowledge) with Reciprocal Rank Fusion.

Dual-Mode Retrieval:
  FAST MODE  — simple factual queries: direct top-k vector, no pool inflation,
               no graph, no reranking. Latency: ~50-150ms.
  FULL MODE  — complex queries: candidate pool + graph context + RRF + cross-encoder.
               Graph context provides entity relationships and community summaries
               that augment vector results for cross-document reasoning.
"""
from typing import List, Dict, Any
from loguru import logger

from ragbox.utils.vector_stores import VectorStore
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.utils.embeddings import EmbeddingProvider
from ragbox.utils.llm_clients import LLMClient
from ragbox.models.queries import Source

_SKIP_RERANK_MIN_SCORE: float = 0.92
_SKIP_RERANK_MIN_GAP: float = 0.15


class RetrievalFusionEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: OptimizedKnowledgeGraph,
        embedding_provider: EmbeddingProvider,
        llm_client: LLMClient,
    ):
        self.vstore = vector_store
        self.kg = knowledge_graph
        self.embeddings = embedding_provider
        self.llm = llm_client

        from ragbox.core.reranker import CrossEncoderReranker

        self.reranker = CrossEncoderReranker()

    async def retrieve(
        self, query: str, top_k: int = 5, fast_mode: bool = False
    ) -> List[Source]:
        """
        Hybrid retrieval with dual-mode dispatch.

        fast_mode=True  → Direct vector top-k. No graph, no reranking.
        fast_mode=False → Full pipeline: vector + graph + RRF + cross-encoder.
        """
        query_emb = await self.embeddings.embed_query(query)

        if fast_mode:
            return await self._retrieve_fast(query, query_emb, top_k)
        else:
            return await self._retrieve_full(query, query_emb, top_k)

    async def _retrieve_fast(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """Pure vector lookup — no graph, no reranking."""
        logger.info(f"FAST retrieval for: {query[:60]}")
        results = await self.vstore.search(query_emb, k=top_k)
        return [
            Source(
                document_id=r.metadata.get("doc_id", "unknown"),
                text=r.content, score=r.score,
            )
            for r in results[:top_k]
        ]

    async def _retrieve_full(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """
        Full pipeline: vector search + graph context + RRF + cross-encoder.

        Key improvement: graph results inject entity source texts and
        relationship context as additional retrieval candidates, giving the
        LLM cross-document information that pure vector search misses.
        """
        import asyncio

        logger.info(f"FULL retrieval for: {query[:60]}")
        candidate_pool_size = max(50, top_k * 10)

        # Run vector search and graph query in parallel
        vector_task = self.vstore.search(query_emb, k=candidate_pool_size)
        graph_task = self.kg.query(query)
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)

        # Merge via RRF
        merged = self._rrf(vector_results, graph_results, k=60)

        # Adaptive rerank skip
        if self._should_skip_rerank(vector_results, graph_results):
            logger.debug("Adaptive rerank skip — high-confidence top result")
            return [
                Source(
                    document_id=item.get("metadata", {}).get("doc_id", "unknown"),
                    text=item.get("content", ""), score=item.get("score", 0.0),
                )
                for item in merged[:top_k]
            ]

        # Cross-encoder reranking on pruned candidates
        pruned = merged[: max(15, top_k * 3)]
        reranked = await self.reranker.rerank(query, pruned, top_k=top_k)

        return [
            Source(
                document_id=item.get("metadata", {}).get("doc_id", "unknown"),
                text=item.get("content", ""),
                score=item.get("cross_encoder_score", item.get("score", 0.0)),
            )
            for item in reranked
        ]

    def _should_skip_rerank(self, vector_results: list, graph_results: Any) -> bool:
        if graph_results and graph_results.synthesized_context:
            return False
        if len(vector_results) < 2:
            return False

        def _score(r):
            return r.score if hasattr(r, "score") else r.get("score", 0.0)

        top = _score(vector_results[0])
        second = _score(vector_results[1])
        return top >= _SKIP_RERANK_MIN_SCORE and (top - second) >= _SKIP_RERANK_MIN_GAP

    def _rrf(
        self, vector_res: list, graph_res: Any, k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion that properly integrates graph context.

        Graph context is split into meaningful chunks (entity descriptions,
        source texts, community summaries) and injected as separate candidates
        so the cross-encoder can score them individually.
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        # Vector results
        for rank, item in enumerate(vector_res):
            id_ = item.id if hasattr(item, "id") else item.get("id", f"vec_{rank}")
            if id_ not in scores:
                scores[id_] = 0.0
                items[id_] = (
                    item
                    if isinstance(item, dict)
                    else {
                        "id": id_,
                        "content": item.content if hasattr(item, "content") else "",
                        "metadata": item.metadata if hasattr(item, "metadata") else {},
                        "score": item.score if hasattr(item, "score") else 0.0,
                    }
                )
            scores[id_] += 1.0 / (k + rank)

        # Graph results — inject as individual context chunks
        if graph_res and graph_res.synthesized_context:
            # Split graph context into meaningful segments for individual scoring
            context_segments = self._split_graph_context(graph_res)
            for idx, segment in enumerate(context_segments):
                g_id = f"graph_ctx_{idx}"
                # Graph context gets a rank boost proportional to its position
                scores[g_id] = 1.0 / (k + idx)
                items[g_id] = {
                    "id": g_id,
                    "content": segment,
                    "metadata": {"doc_id": "knowledge_graph", "source": "graph"},
                    "score": 1.0 / (k + idx),
                }

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = []
        for id_ in sorted_ids:
            item = items[id_]
            item["score"] = scores[id_]
            result.append(item)
        return result

    def _split_graph_context(self, graph_res: Any) -> List[str]:
        """
        Split graph context into coherent segments for individual RRF scoring.
        Each entity block and community summary becomes a separate candidate.
        """
        full_context = graph_res.synthesized_context
        if not full_context:
            return []

        # Split on double newlines (each entity/community block)
        segments = [s.strip() for s in full_context.split("\n\n") if s.strip()]

        # Merge very short segments and cap very long ones
        result = []
        buffer = ""
        for seg in segments:
            if len(seg) < 50 and buffer:
                buffer += "\n" + seg
            elif len(buffer) + len(seg) < 1500:
                buffer = (buffer + "\n" + seg).strip() if buffer else seg
            else:
                if buffer:
                    result.append(buffer)
                buffer = seg
        if buffer:
            result.append(buffer)

        return result[:10]  # Cap at 10 segments
