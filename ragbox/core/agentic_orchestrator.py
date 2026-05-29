"""
Layer 6: AGENTIC ORCHESTRATION
Query classification, dynamic routing, multi-hop decomposition, and synthesis.

Key designs:
  1. Speculative Parallel Execution — vector search + classifier fire simultaneously.
  2. Heuristic Pre-classifier — zero-cost regex for obvious query types.
  3. Graph+Vector Fusion — graph context is combined with vector results in the prompt.
  4. Multi-hop Decomposition — complex queries are split into sub-queries.
"""
import re
import time
import asyncio
from typing import Any, Optional
from loguru import logger

from ragbox.models.queries import Answer, RAGStrategy
from ragbox.core.retrieval_fusion import RetrievalFusionEngine
from ragbox.utils.llm_clients import LLMClient


# ---------------------------------------------------------------------------
# Heuristic pre-classifier — zero LLM cost, zero network latency
# ---------------------------------------------------------------------------
_GRAPH_PATTERNS = re.compile(
    r"\b(who does .+ report to|how does .+ relate|what connects|"
    r"relationship between|links? between|responsible for both|"
    r"who (?:manages|owns|leads|oversees) .+ (?:and|who)|"
    r"what (?:went wrong|happened|caused) .+ (?:and|who)|"
    r"how .+ (?:affect|impact|influence|relate to) .+|"
    r"what is the (?:connection|link|relationship) between)\b",
    re.IGNORECASE,
)

_MULTI_HOP_PATTERNS = re.compile(
    r"\b((?:and\s+(?:what|how|who|which))|"
    r"(?:what .{5,30} and .{5,30}\?)|"
    r"(?:how many .{5,30} and .{5,30}))\b",
    re.IGNORECASE,
)

_MULTI_QUERY_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference between|similarities|"
    r"pros and cons|trade.?off|how do i .{5,40} step|walk me through)\b",
    re.IGNORECASE,
)

_FACTUAL_SIMPLE = re.compile(
    r"^(what is|what are|how many|how much|when|where|who is|"
    r"what was|what were|list|define|describe)\b.{0,80}$",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> Optional[RAGStrategy]:
    """Zero-cost pre-classifier. Returns a strategy or None for LLM fallback."""
    q = query.strip()

    if _GRAPH_PATTERNS.search(q):
        logger.debug("Heuristic: GRAPH (relationship signal detected)")
        return RAGStrategy.GRAPH

    if _MULTI_HOP_PATTERNS.search(q):
        logger.debug("Heuristic: MULTI_QUERY (multi-hop signal detected)")
        return RAGStrategy.MULTI_QUERY

    if _MULTI_QUERY_PATTERNS.search(q):
        logger.debug("Heuristic: MULTI_QUERY (comparison signal detected)")
        return RAGStrategy.MULTI_QUERY

    if _FACTUAL_SIMPLE.match(q) and len(q.split()) <= 12:
        logger.debug("Heuristic: VECTOR (short factual query)")
        return RAGStrategy.VECTOR

    return None


class AgenticOrchestrator:
    def __init__(
        self,
        retriever: RetrievalFusionEngine,
        llm_client: LLMClient,
        document_processor: Any,
        knowledge_graph: Any,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.doc_processor = document_processor
        self.kg = knowledge_graph

    async def _classify_query(self, query: str) -> RAGStrategy:
        """LLM-based query classifier. Only called when heuristic is uncertain."""
        schema = {
            "strategy": "vector | graph | multi_query",
            "reasoning": "string",
        }
        prompt = (
            f"Classify this query into a retrieval strategy:\n"
            f"- 'vector': Simple factual lookup (\"What is X?\", \"How many Y?\")\n"
            f"- 'graph': Questions about relationships, connections, or cross-topic "
            f"reasoning (\"How does X relate to Y?\", \"Who manages X?\")\n"
            f"- 'multi_query': Complex queries needing multiple perspectives, "
            f"comparisons, or multi-step reasoning (\"Compare A and B\", "
            f"\"What is X and how does it affect Y?\")\n\n"
            f"Query: {query}"
        )
        try:
            res = await self.llm.agenerate_structured(prompt, schema)
            strategy_str = res.get("strategy", "vector").lower().strip()
            mapping = {
                "graph": RAGStrategy.GRAPH,
                "multi_query": RAGStrategy.MULTI_QUERY,
            }
            return mapping.get(strategy_str, RAGStrategy.VECTOR)
        except Exception as e:
            logger.warning(f"Classification failed: {e}. Defaulting to VECTOR.")
            return RAGStrategy.VECTOR

    async def _expand_query(self, query: str) -> list[str]:
        """Expand one query into multiple semantic variations."""
        prompt = (
            f"Generate 3 distinct semantic variations of this query to improve "
            f"search recall. Return ONLY the queries separated by newlines.\n"
            f"Query: {query}"
        )
        try:
            res = await self.llm.agenerate(
                prompt, system="You are an expert search query generator."
            )
            queries = [q.strip("- \t1234567890.") for q in res.split("\n") if q.strip()]
            queries = [q for q in queries if q and len(q) > 5]
            return queries[:3] if queries else [query]
        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}")
            return [query]

    async def _decompose_query(self, query: str) -> list[str]:
        """
        Decompose a multi-hop query into atomic sub-queries.
        E.g. "How many engineers and what % are senior?" →
             ["How many engineers?", "What percentage are senior?"]
        """
        prompt = (
            f"Break this complex question into 2-3 simple, self-contained "
            f"sub-questions. Each sub-question should be answerable independently.\n"
            f"Return ONLY the sub-questions, one per line.\n\n"
            f"Question: {query}"
        )
        try:
            res = await self.llm.agenerate(
                prompt,
                system="You decompose complex questions into simple sub-questions.",
            )
            subs = [q.strip("- \t1234567890.") for q in res.split("\n") if q.strip()]
            subs = [q for q in subs if q and len(q) > 5]
            return subs[:4] if subs else [query]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]

    # ── Main Execute ──────────────────────────────────────────────────────

    async def execute(
        self, query_text: str, force_strategy: Optional[RAGStrategy] = None
    ) -> Answer:
        """
        Execute the end-to-end RAG pipeline with speculative parallel execution.
        """
        start_time = time.time()

        if force_strategy:
            strategy = force_strategy
            speculative_sources = None
            heuristic_was_vector = False
        else:
            heuristic = _heuristic_classify(query_text)

            if heuristic is not None:
                strategy = heuristic
                speculative_sources = None
                heuristic_was_vector = heuristic == RAGStrategy.VECTOR
                logger.info(
                    f"Heuristic: '{query_text[:50]}' → {strategy.name} "
                    f"(LLM classify skipped)"
                )
            else:
                logger.info(
                    f"Heuristic uncertain for '{query_text[:50]}' — "
                    f"speculative parallel execution"
                )
                (strategy, speculative_sources) = await asyncio.gather(
                    self._classify_query(query_text),
                    self.retriever.retrieve(query_text),
                )
                heuristic_was_vector = False
                logger.info(f"Speculative done — classifier chose {strategy.name}")

        logger.info(f"Executing '{query_text[:60]}' with strategy {strategy.name}")

        if strategy == RAGStrategy.AGENTIC:
            answer = await self._execute_agentic(query_text)
        elif strategy == RAGStrategy.GRAPH:
            answer = await self._execute_graph(query_text)
        elif strategy == RAGStrategy.MULTI_QUERY:
            answer = await self._execute_multi_query(query_text)
        else:
            answer = await self._execute_vector(
                query_text,
                prefetched_sources=speculative_sources,
                fast_mode=heuristic_was_vector,
            )

        answer.execution_time_ms = (time.time() - start_time) * 1000
        return answer

    async def stream_execute(self, query_text: str):
        """Stream the answer token-by-token."""
        heuristic = _heuristic_classify(query_text)
        strategy = heuristic if heuristic is not None else await self._classify_query(query_text)

        logger.info(f"Streaming '{query_text[:60]}' with strategy {strategy.name}")

        if strategy == RAGStrategy.MULTI_QUERY:
            sources = await self._gather_multi_query_sources(query_text)
        elif strategy == RAGStrategy.GRAPH:
            sources = await self._gather_graph_sources(query_text)
        else:
            sources = await self.retriever.retrieve(query_text)

        context = "\n\n---\n\n".join([s.text for s in sources])
        prompt = (
            f"Answer the query based ONLY on the following context.\n\n"
            f"Context:\n{context}\n\nQuery: {query_text}"
        )
        system = "You are an expert Q&A engine. Be concise and accurate."

        async for chunk in self.llm.astream(prompt, system=system):
            yield chunk

    # ── Strategy Executors ────────────────────────────────────────────────

    async def _execute_vector(
        self, query: str, prefetched_sources=None, fast_mode: bool = False
    ) -> Answer:
        """
        Vector path with precision prompting for factual queries.
        """
        if prefetched_sources is not None:
            sources = prefetched_sources
        elif fast_mode:
            sources = await self.retriever.retrieve(query, top_k=5, fast_mode=True)
        else:
            sources = await self.retriever.retrieve(query)

        context = "\n\n---\n\n".join([s.text for s in sources])

        if fast_mode:
            prompt = (
                f"Read the context below and extract the single most relevant fact "
                f"that directly answers the question. "
                f"Answer in ONE concise sentence. Do not add explanation.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            system = "You are a precise fact extractor. One sentence only."
        else:
            prompt = (
                f"Answer the query based ONLY on the following context.\n\n"
                f"Context:\n{context}\n\nQuery: {query}"
            )
            system = "You are an expert Q&A engine. Be concise and accurate."

        response = await self.llm.agenerate(prompt, system=system)
        return Answer(
            query=query, content=response,
            sources=sources, strategy_used=RAGStrategy.VECTOR,
        )

    async def _execute_graph(self, query: str) -> Answer:
        """
        Graph path — combines vector results WITH graph context.

        The key insight: don't choose between vector and graph.
        Use BOTH. Vector provides document chunks, graph provides
        entity relationships and cross-document connections.
        """
        sources = await self._gather_graph_sources(query)
        context = "\n\n---\n\n".join([s.text for s in sources])

        prompt = (
            f"Answer this query using ALL of the following context. The context "
            f"includes both document excerpts and knowledge graph information "
            f"(entity relationships, descriptions, and community summaries). "
            f"Use the graph information to reason about relationships and "
            f"connections between entities.\n\n"
            f"Context:\n{context}\n\nQuery: {query}"
        )
        response = await self.llm.agenerate(
            prompt,
            system=(
                "You are a graph-aware reasoning engine. Use entity relationships "
                "and document context together to give precise answers about "
                "connections, hierarchies, and cross-document relationships."
            ),
        )
        return Answer(
            query=query, content=response,
            sources=sources, strategy_used=RAGStrategy.GRAPH,
        )

    async def _execute_multi_query(self, query: str) -> Answer:
        """
        Multi-query path with query decomposition for multi-hop reasoning.

        1. Decompose query into sub-queries
        2. Also expand with semantic variations
        3. Retrieve for all sub-queries + expansions
        4. Deduplicate and synthesize
        """
        sources = await self._gather_multi_query_sources(query)
        context = "\n\n---\n\n".join([s.text for s in sources])

        prompt = (
            f"Answer the following complex question comprehensively using ALL "
            f"the context below. The context was gathered from multiple search "
            f"perspectives to ensure completeness. Synthesize information from "
            f"different parts of the context to build a complete answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        response = await self.llm.agenerate(
            prompt,
            system=(
                "You are an expert at synthesizing information from multiple "
                "sources. Combine facts from different parts of the context to "
                "answer complex, multi-part questions thoroughly."
            ),
        )
        return Answer(
            query=query, content=response,
            sources=sources, strategy_used=RAGStrategy.MULTI_QUERY,
        )

    async def _execute_agentic(self, query: str) -> Answer:
        """
        Agentic path — iterative retrieval with reasoning.
        Uses structured search-then-reason loops with actual context accumulation.
        """
        accumulated_context = []
        all_sources = []
        max_steps = 4

        # Step 0: Initial broad retrieval
        initial_sources = await self.retriever.retrieve(query, top_k=5)
        all_sources.extend(initial_sources)
        accumulated_context.extend([s.text for s in initial_sources])

        for step in range(max_steps):
            context_so_far = "\n---\n".join(accumulated_context[-10:])
            prompt = (
                f"You are investigating this question: {query}\n\n"
                f"Context gathered so far:\n{context_so_far}\n\n"
                f"Based on this context, do you have enough information to "
                f"answer the question fully?\n"
                f"If YES, respond with: ANSWER: <your complete answer>\n"
                f"If NO, respond with: SEARCH: <specific search term to find "
                f"missing information>"
            )
            action = await self.llm.agenerate(
                prompt,
                system="You are a research agent. Decide if you need more info or can answer.",
            )

            if "ANSWER:" in action:
                final_ans = action.split("ANSWER:", 1)[-1].strip()
                return Answer(
                    query=query, content=final_ans,
                    sources=all_sources, strategy_used=RAGStrategy.AGENTIC,
                )
            elif "SEARCH:" in action:
                search_term = action.split("SEARCH:", 1)[-1].strip()
                if search_term:
                    step_sources = await self.retriever.retrieve(search_term, top_k=3)
                    all_sources.extend(step_sources)
                    accumulated_context.extend([s.text for s in step_sources])
            else:
                break

        # Fallback: synthesize from everything we gathered
        final_context = "\n\n---\n\n".join(
            [s.text for s in all_sources[:15]]
        )
        response = await self.llm.agenerate(
            f"Answer this question using the context below.\n\n"
            f"Context:\n{final_context}\n\nQuestion: {query}",
            system="You are an expert Q&A engine. Be thorough and accurate.",
        )
        return Answer(
            query=query, content=response,
            sources=all_sources, strategy_used=RAGStrategy.AGENTIC,
        )

    # ── Source Gathering Helpers ──────────────────────────────────────────

    async def _gather_graph_sources(self, query: str) -> list:
        """Retrieve sources combining vector search with graph context."""
        # Get graph context directly from the knowledge graph
        graph_result = await self.kg.query(query)
        graph_context = graph_result.synthesized_context if graph_result else ""

        # Also get vector results
        sources = await self.retriever.retrieve(query, top_k=5)

        # Inject graph context as a high-priority source if non-trivial
        if graph_context and len(graph_context) > 50:
            from ragbox.models.queries import Source

            graph_source = Source(
                document_id="knowledge_graph",
                text=graph_context,
                score=1.0,
            )
            # Place graph context first, then vector results
            sources = [graph_source] + [s for s in sources if s.text != graph_context]

        return sources[:8]

    async def _gather_multi_query_sources(self, query: str) -> list:
        """Retrieve sources from decomposed sub-queries + semantic expansions."""
        # Run decomposition and expansion in parallel
        decomposed, expanded = await asyncio.gather(
            self._decompose_query(query),
            self._expand_query(query),
        )

        all_queries = [query] + decomposed + expanded
        # Deduplicate queries
        seen_q = {query}
        unique_queries = [query]
        for q in decomposed + expanded:
            if q.lower() not in seen_q:
                seen_q.add(q.lower())
                unique_queries.append(q)

        logger.info(f"Multi-query retrieval with {len(unique_queries)} queries")

        # Retrieve for all queries (with concurrency limit)
        all_sources = []
        sem = asyncio.Semaphore(3)

        async def _retrieve_one(q):
            async with sem:
                return await self.retriever.retrieve(q, top_k=3)

        results = await asyncio.gather(*[_retrieve_one(q) for q in unique_queries])
        for r in results:
            all_sources.extend(r)

        # Deduplicate by text content
        seen = set()
        unique_sources = []
        for s in all_sources:
            text_key = s.text[:200]  # Use prefix as dedup key
            if text_key not in seen:
                seen.add(text_key)
                unique_sources.append(s)

        # Sort by score descending
        unique_sources.sort(key=lambda s: s.score, reverse=True)
        return unique_sources[:12]
