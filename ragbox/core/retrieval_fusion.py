"""
Layer 5: RETRIEVAL FUSION
Dense vectors + SQLite FTS5 sparse search + graph context with weighted RRF.

Retrieval modes:
  FAST             — dense+sparse RRF, no graph or reranking.
  TRI_MODAL_FAST   — dense+sparse+graph RRF, no reranking.
  FULL             — dense+sparse+graph RRF + rank-fused cross-encoder.
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional
from loguru import logger

from ragbox.utils.vector_stores import VectorStore
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.utils.embeddings import EmbeddingProvider
from ragbox.utils.llm_clients import LLMClient
from ragbox.models.queries import Source

_SKIP_RERANK_MIN_SCORE: float = 0.92
_SKIP_RERANK_MIN_GAP: float = 0.15
_DENSE_RRF_WEIGHT: float = 1.0
_SPARSE_RRF_WEIGHT: float = 0.85
_GRAPH_RRF_WEIGHT: float = 0.65
_RANK_FUSION_K: int = 60
_VALID_MODES = frozenset({"fast", "tri_modal_fast", "full"})


class RetrievalFusionEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: OptimizedKnowledgeGraph,
        embedding_provider: EmbeddingProvider,
        llm_client: LLMClient,
        sparse_search_fn: Optional[Callable[[str, int], list]] = None,
    ):
        self.vstore = vector_store
        self.kg = knowledge_graph
        self.embeddings = embedding_provider
        self.llm = llm_client
        self.active_revision_resolver: Optional[Callable[[str, str], bool]] = None
        self.sparse_search_fn = sparse_search_fn

        from ragbox.core.reranker import CrossEncoderReranker

        self.reranker = CrossEncoderReranker()

    def set_active_revision_resolver(
        self, resolver: Callable[[str, str], bool]
    ) -> None:
        self.active_revision_resolver = resolver

    def set_sparse_search(self, search_fn: Callable[[str, int], list]) -> None:
        self.sparse_search_fn = search_fn

    async def _search_sparse(self, query: str, limit: int) -> list:
        if self.sparse_search_fn is None:
            return []
        return await asyncio.to_thread(self.sparse_search_fn, query, limit)

    def _active_only(self, results: list) -> list:
        if self.active_revision_resolver is None:
            return results
        active = []
        for result in results:
            metadata = (
                result.metadata
                if hasattr(result, "metadata")
                else result.get("metadata", {})
            )
            source_path = metadata.get("source_path", "")
            revision_id = metadata.get("revision_id", metadata.get("doc_id", ""))
            if self.active_revision_resolver(source_path, revision_id):
                active.append(result)
        return active

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fast_mode: bool = False,
        mode: Optional[str] = None,
    ) -> List[Source]:
        """Retrieve with an explicit mode or the legacy ``fast_mode`` alias."""
        selected_mode = self._resolve_mode(fast_mode=fast_mode, mode=mode)
        query_emb = await self._embed_query(query)

        if selected_mode == "fast":
            return await self._retrieve_fast(query, query_emb, top_k)
        if selected_mode == "tri_modal_fast":
            return await self._retrieve_tri_modal_fast(query, query_emb, top_k)
        return await self._retrieve_full(query, query_emb, top_k)

    @staticmethod
    def _resolve_mode(*, fast_mode: bool, mode: Optional[str]) -> str:
        if mode is None:
            return "fast" if fast_mode else "full"
        if mode not in _VALID_MODES:
            valid = ", ".join(sorted(_VALID_MODES))
            raise ValueError(
                f"Unknown retrieval mode {mode!r}; expected one of: {valid}"
            )
        if fast_mode and mode != "fast":
            raise ValueError("fast_mode=True conflicts with an explicit non-fast mode")
        return mode

    async def _retrieve_fast(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """Dense+sparse lookup — no graph and no cross-encoder."""
        logger.info(f"FAST retrieval for: {query[:60]}")
        candidate_pool_size = max(top_k * 4, top_k + 10)
        results, sparse_results = await asyncio.gather(
            self._search_dense(query_emb, candidate_pool_size),
            self._search_sparse(query, candidate_pool_size),
        )
        results = self._active_only(results)
        merged = self._rrf(results, None, sparse_results)
        return self._to_sources(merged[:top_k])

    async def _retrieve_tri_modal_fast(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """Dense+sparse+graph RRF without cross-encoder reranking."""
        logger.info(f"TRI_MODAL_FAST retrieval for: {query[:60]}")
        candidate_pool_size = max(top_k * 4, top_k + 10)
        vector_results, graph_results, sparse_results = await self._tri_modal_search(
            query, query_emb, candidate_pool_size
        )
        merged = self._rrf(vector_results, graph_results, sparse_results, k=60)
        return self._to_sources(self._select_top_k(merged, top_k))

    async def _retrieve_full(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """
        Full pipeline: vector search + graph context + RRF + cross-encoder.

        Key improvement: graph results inject entity source texts and
        relationship context as additional retrieval candidates, giving the
        LLM cross-document information that pure vector search misses.
        """
        logger.info(f"FULL retrieval for: {query[:60]}")
        candidate_pool_size = max(50, top_k * 10)

        vector_results, graph_results, sparse_results = await self._tri_modal_search(
            query, query_emb, candidate_pool_size
        )

        # Merge via RRF
        merged = self._rrf(vector_results, graph_results, sparse_results, k=60)

        # Adaptive rerank skip
        if self._should_skip_rerank(vector_results, graph_results, sparse_results):
            logger.debug("Adaptive rerank skip — high-confidence top result")
            return self._to_sources(self._select_top_k(merged, top_k))

        # Cross-encoder reranking on pruned candidates
        pruned = self._prune_for_rerank(merged, top_k)
        reranked = await self._rerank_candidates(query, pruned)
        blended = self._blend_rank_scores(reranked)
        return self._to_sources(
            self._select_top_k(blended, top_k), score_key="blended_score"
        )

    async def _tri_modal_search(
        self, query: str, query_emb: list, candidate_pool_size: int
    ) -> tuple[list, Any, list]:
        vector_results, graph_results, sparse_results = await asyncio.gather(
            self._search_dense(query_emb, candidate_pool_size),
            self._query_graph(query),
            self._search_sparse(query, candidate_pool_size),
        )
        return self._active_only(vector_results), graph_results, sparse_results

    async def _embed_query(self, query: str) -> list:
        return await self.embeddings.embed_query(query)

    async def _search_dense(self, query_emb: list, limit: int) -> list:
        return await self.vstore.search(query_emb, k=limit)

    async def _query_graph(self, query: str) -> Any:
        return await self.kg.query(query)

    async def _rerank_candidates(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return await self.reranker.rerank(query, candidates, top_k=len(candidates))

    @staticmethod
    def _to_sources(
        items: List[Dict[str, Any]], score_key: str = "score"
    ) -> List[Source]:
        sources: List[Source] = []
        for item in items:
            metadata = item.get("metadata", {})
            document_id = metadata.get("doc_id", "unknown")
            is_graph = metadata.get("source") == "graph"
            sources.append(
                Source(
                    document_id=document_id,
                    chunk_id=item.get("id"),
                    source_path=metadata.get("source_path"),
                    revision_id=metadata.get("revision_id")
                    or (None if is_graph else document_id),
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                    text=item.get("content", ""),
                    score=item.get(score_key, item.get("score", 0.0)),
                )
            )
        return sources

    @staticmethod
    def _select_top_k(ranked: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Return at most one synthetic graph block so documents keep their slots."""
        if top_k <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        graph_count = 0
        for item in ranked:
            is_graph = item.get("metadata", {}).get("source") == "graph"
            if is_graph and graph_count >= 1:
                continue
            selected.append(item)
            graph_count += int(is_graph)
            if len(selected) == top_k:
                break
        return selected

    @staticmethod
    def _blend_rank_scores(
        reranked: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse zero-based RRF and cross-encoder ranks without score calibration."""
        for fallback_ce_rank, item in enumerate(reranked):
            rrf_rank = int(item.get("rrf_rank", fallback_ce_rank))
            ce_rank = int(item.get("ce_rank", fallback_ce_rank))
            item["rrf_rank"] = rrf_rank
            item["ce_rank"] = ce_rank
            item["blended_score"] = (1.0 / (_RANK_FUSION_K + rrf_rank)) + (
                1.0 / (_RANK_FUSION_K + ce_rank)
            )
        return sorted(
            reranked,
            key=lambda item: (-item["blended_score"], item["rrf_rank"]),
        )

    def _should_skip_rerank(
        self,
        vector_results: list,
        graph_results: Any,
        sparse_results: tuple | list = (),
    ) -> bool:
        if vector_results and sparse_results:
            vector_id = (
                vector_results[0].id
                if hasattr(vector_results[0], "id")
                else vector_results[0].get("id")
            )
            sparse_id = sparse_results[0].get("id")
            if vector_id and vector_id == sparse_id:
                return True
        if graph_results and graph_results.synthesized_context:
            return False
        if len(vector_results) < 2:
            return False

        def _score(r):
            return r.score if hasattr(r, "score") else r.get("score", 0.0)

        top = _score(vector_results[0])
        second = _score(vector_results[1])
        return top >= _SKIP_RERANK_MIN_SCORE and (top - second) >= _SKIP_RERANK_MIN_GAP

    @staticmethod
    def _prune_for_rerank(
        merged: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Keep an eight-candidate pool with a bounded graph quota."""
        del top_k
        limit = 8
        graph_candidates = [
            item for item in merged if item.get("metadata", {}).get("source") == "graph"
        ]
        if not graph_candidates:
            return merged[:limit]

        graph_quota = min(2, len(graph_candidates), max(1, limit // 4))
        non_graph = [
            item for item in merged if item.get("metadata", {}).get("source") != "graph"
        ]
        return non_graph[: limit - graph_quota] + graph_candidates[:graph_quota]

    def _rrf(
        self,
        vector_res: list,
        graph_res: Any,
        sparse_res: tuple | list = (),
        k: int = 60,
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
            scores[id_] += _DENSE_RRF_WEIGHT / (k + rank + 1)

        # Sparse FTS5 results
        for rank, item in enumerate(sparse_res):
            id_ = item.get("id", f"sparse_{rank}")
            if id_ not in scores:
                scores[id_] = 0.0
                items[id_] = item
            scores[id_] += _SPARSE_RRF_WEIGHT / (k + rank + 1)

        # Graph results — inject as individual context chunks
        if graph_res and graph_res.synthesized_context:
            # Split graph context into meaningful segments for individual scoring
            context_segments = self._split_graph_context(graph_res)
            for idx, segment in enumerate(context_segments):
                g_id = f"graph_ctx_{idx}"
                # Graph context gets a rank boost proportional to its position
                scores[g_id] = _GRAPH_RRF_WEIGHT / (k + idx + 1)
                items[g_id] = {
                    "id": g_id,
                    "content": segment,
                    "metadata": {"doc_id": "knowledge_graph", "source": "graph"},
                    "score": _GRAPH_RRF_WEIGHT / (k + idx + 1),
                }

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result: List[Dict[str, Any]] = []
        for id_ in sorted_ids:
            item = items[id_]
            item["score"] = scores[id_]
            item["rrf_rank"] = len(result)
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
