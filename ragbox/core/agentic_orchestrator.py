"""
Layer 6: AGENTIC ORCHESTRATION
Query classification, dynamic routing, ReAct planning, and synthesis.

Key design: Speculative Parallel Execution.
  - Vector search + LLM classifier fire simultaneously.
  - If classifier returns VECTOR, results are already waiting — zero overhead.
  - A zero-cost heuristic pre-classifier runs first to skip the LLM call
    entirely for obviously simple or obviously relational queries.
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
    r"who (?:manages|owns|leads|oversees) .+ and who|"
    r"what (?:went wrong|happened|caused) .+ and who)\b",
    re.IGNORECASE,
)

_MULTI_QUERY_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference between|similarities|"
    r"pros and cons|trade.?off|how do i .{5,40} step|walk me through)\b",
    re.IGNORECASE,
)

# Simple factual: short, starts with what/how many/when/where, no cross-doc signals
_FACTUAL_SIMPLE = re.compile(
    r"^(what is|what are|how many|how much|when|where|who is|"
    r"what was|what were|list|define|describe)\b.{0,80}$",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> Optional[RAGStrategy]:
    """
    Zero-cost pre-classifier. Returns a strategy when confidence is high,
    or None to fall back to the LLM classifier.
    """
    q = query.strip()

    if _GRAPH_PATTERNS.search(q):
        logger.debug("Heuristic: GRAPH (relationship signal detected)")
        return RAGStrategy.GRAPH

    if _MULTI_QUERY_PATTERNS.search(q):
        logger.debug("Heuristic: MULTI_QUERY (comparison signal detected)")
        return RAGStrategy.MULTI_QUERY

    # Short factual queries with no cross-document signals
    if _FACTUAL_SIMPLE.match(q) and len(q.split()) <= 12:
        logger.debug("Heuristic: VECTOR (short factual query, skipping LLM classify)")
        return RAGStrategy.VECTOR

    return None  # Uncertain — hand off to LLM classifier


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
            "strategy": "vector | agentic | graph | multi_query",
            "reasoning": "string",
        }
        prompt = f"""
        Classify this query into one of the following retrieval strategies based on intent:
        - 'vector': General semantic search, asking "What is X?"
        - 'graph': Relationships, connections ("How does X relate to Y?")
        - 'multi_query': Broad comparisons, multi-step ("Compare A and B", "How do I deploy?")
        - 'agentic': Complex reasoning requiring outside tool usage or sequence of thoughts.

        Query: {query}
        """

        try:
            res = await self.llm.agenerate_structured(prompt, schema)
            strategy_str = res.get("strategy", "vector").lower()
            if strategy_str == "graph":
                return RAGStrategy.GRAPH
            elif strategy_str == "agentic":
                return RAGStrategy.AGENTIC
            elif strategy_str == "multi_query":
                return RAGStrategy.MULTI_QUERY
            return RAGStrategy.VECTOR
        except Exception as e:
            logger.warning(f"Classification failed: {e}. Defaulting to VECTOR.")
            return RAGStrategy.VECTOR

    async def _expand_query(self, query: str) -> list[str]:
        """Expand one query into multiple semantic variations."""
        prompt = f"""
        Generate 3 distinct semantic variations of this query to improve search recall. 
        Return ONLY the queries separated by newlines.
        Query: {query}
        """
        try:
            res = await self.llm.agenerate(
                prompt, system="You are an expert search query generator."
            )
            queries = [q.strip("- \t1234567890.") for q in res.split("\n") if q.strip()]
            queries = [q for q in queries if q]
            if not queries:
                return [query]
            return queries[:3]
        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}")
            return [query]

    async def execute(
        self, query_text: str, force_strategy: Optional[RAGStrategy] = None
    ) -> Answer:
        """
        Execute the end-to-end RAG pipeline with speculative parallel execution.

        Strategy:
          1. Run heuristic pre-classifier (zero cost).
          2. If heuristic is certain → skip LLM classify entirely.
          3. If heuristic is uncertain → fire vector search + LLM classifier
             IN PARALLEL using asyncio.gather (speculative execution).
          4. Route answer using whichever strategy was chosen.
             On VECTOR path: results are already waiting from step 3 — zero overhead.
        """
        start_time = time.time()

        if force_strategy:
            strategy = force_strategy
            speculative_sources = None
            heuristic_was_vector = False
        else:
            # ── Phase 1: heuristic (free) ────────────────────────────────────
            heuristic = _heuristic_classify(query_text)

            if heuristic is not None:
                # High confidence — skip LLM classifier entirely
                strategy = heuristic
                speculative_sources = None
                heuristic_was_vector = heuristic == RAGStrategy.VECTOR
                logger.info(
                    f"Heuristic classified '{query_text[:50]}' as {strategy.name} "
                    f"(LLM classify skipped)"
                )
            else:
                # ── Phase 2: speculative parallel execution ──────────────────
                # Fire vector search AND LLM classifier simultaneously.
                # If strategy comes back VECTOR, results are already ready.
                logger.info(
                    f"Heuristic uncertain for '{query_text[:50]}' — "
                    f"launching speculative parallel execution"
                )
                (strategy, speculative_sources) = await asyncio.gather(
                    self._classify_query(query_text),
                    self.retriever.retrieve(query_text),
                )
                heuristic_was_vector = False
                logger.info(
                    f"Speculative results ready — classifier chose {strategy.name}"
                )

        logger.info(f"Executing '{query_text[:60]}' with strategy {strategy.name}")

        if strategy == RAGStrategy.AGENTIC:
            answer = await self._execute_agentic(query_text)
        elif strategy == RAGStrategy.GRAPH:
            answer = await self._execute_graph(query_text)
        elif strategy == RAGStrategy.MULTI_QUERY:
            answer = await self._execute_multi_query(query_text)
        else:
            # VECTOR — reuse speculative results if available
            # Pass fast_mode=True if heuristic confidently identified this as simple factual
            answer = await self._execute_vector(
                query_text,
                prefetched_sources=speculative_sources,
                fast_mode=heuristic_was_vector,
            )

        answer.execution_time_ms = (time.time() - start_time) * 1000
        return answer

    async def stream_execute(self, query_text: str):
        """Stream the answer token-by-token. Retrieves context first, then streams."""
        heuristic = _heuristic_classify(query_text)
        if heuristic is not None:
            strategy = heuristic
        else:
            strategy = await self._classify_query(query_text)

        logger.info(
            f"Streaming query '{query_text[:60]}' with strategy {strategy.name}"
        )

        if strategy == RAGStrategy.MULTI_QUERY:
            expanded_queries = await self._expand_query(query_text)
            all_sources = []
            for q in [query_text] + expanded_queries:
                q_sources = await self.retriever.retrieve(q, top_k=3)
                all_sources.extend(q_sources)
            seen = set()
            sources = []
            for s in all_sources:
                if s.text not in seen:
                    seen.add(s.text)
                    sources.append(s)
            sources = sources[:10]
        else:
            sources = await self.retriever.retrieve(query_text)

        context = "\n\n---\n\n".join([s.text for s in sources])
        prompt = f"Answer the query based ONLY on the following context.\n\nContext:\n{context}\n\nQuery: {query_text}"
        system = "You are an expert Q&A engine. Be concise and accurate."

        async for chunk in self.llm.astream(prompt, system=system):
            yield chunk

    async def _execute_vector(
        self, query: str, prefetched_sources=None, fast_mode: bool = False
    ) -> Answer:
        """
        Vector path — two sub-modes:

        fast_mode=True  (heuristic-VECTOR): Direct top-5 vector, no reranking.
                        Uses a precision *extraction* prompt: one-sentence fact extract.
                        Produces concise answers that score higher on semantic similarity.

        fast_mode=False (LLM-classified or speculative): Full pipeline via retriever.
                        Uses an open-ended generation prompt for richer answers.
        """
        if prefetched_sources is not None:
            sources = prefetched_sources
        elif fast_mode:
            # Fast path — lightweight retrieval, no reranking
            sources = await self.retriever.retrieve(query, top_k=5, fast_mode=True)
        else:
            sources = await self.retriever.retrieve(query)

        context = "\n\n---\n\n".join([s.text for s in sources])

        if fast_mode:
            # Precision extraction prompt — designed for concise factual answers
            # that closely match ground-truth phrasing
            prompt = (
                f"Read the context below and extract the single most relevant fact "
                f"that directly answers the question. "
                f"Answer in ONE concise sentence. Do not add explanation.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            system = "You are a precise fact extractor. Extract the exact answer. One sentence only."
        else:
            prompt = (
                f"Answer the query based ONLY on the following context.\n\n"
                f"Context:\n{context}\n\nQuery: {query}"
            )
            system = "You are an expert Q&A engine. Be concise and accurate."

        response = await self.llm.agenerate(prompt, system=system)

        return Answer(
            query=query,
            content=response,
            sources=sources,
            strategy_used=RAGStrategy.VECTOR,
        )

    async def _execute_multi_query(self, query: str) -> Answer:
        expanded_queries = await self._expand_query(query)
        logger.info(f"Expanded initial query into: {expanded_queries}")

        all_sources = []
        for q in [query] + expanded_queries:
            q_sources = await self.retriever.retrieve(q, top_k=3)
            all_sources.extend(q_sources)

        seen = set()
        unique_sources = []
        for s in all_sources:
            if s.text not in seen:
                seen.add(s.text)
                unique_sources.append(s)
        unique_sources = unique_sources[:10]

        context = "\n\n---\n\n".join([s.text for s in unique_sources])
        prompt = (
            f"Answer the query comprehensively based ONLY on the following context "
            f"derived from multi-query expansion.\n\nContext:\n{context}\n\nQuery: {query}"
        )
        response = await self.llm.agenerate(
            prompt,
            system="You are an expert Q&A engine analyzing across multiple perspectives.",
        )

        return Answer(
            query=query,
            content=response,
            sources=unique_sources,
            strategy_used=RAGStrategy.MULTI_QUERY,
        )

    async def _execute_graph(self, query: str) -> Answer:
        sources = await self.retriever.retrieve(query)
        context = "\n\n---\n\n".join([s.text for s in sources])

        prompt = (
            f"Answer this query using the graph summaries and retrieved text provided below:"
            f"\n\nContext:\n{context}\n\nQuery: {query}"
        )
        response = await self.llm.agenerate(
            prompt, system="You are a graph-aware reasoning agent."
        )

        return Answer(
            query=query,
            content=response,
            sources=sources,
            strategy_used=RAGStrategy.GRAPH,
        )

    async def _execute_agentic(self, query: str) -> Answer:
        history = []
        max_steps = 3
        sources = []

        for step in range(max_steps):
            prompt = f"Query: {query}\nHistory: {history}\n\nAction (SEARCH <term> or ANSWER <final answer>):"
            action = await self.llm.agenerate(
                prompt, system="You are a ReAct agent. You can SEARCH or ANSWER."
            )

            if "ANSWER" in action:
                final_ans = action.split("ANSWER")[-1].strip(": \n")
                return Answer(
                    query=query,
                    content=final_ans,
                    sources=sources,
                    strategy_used=RAGStrategy.AGENTIC,
                )
            elif "SEARCH" in action:
                search_term = action.split("SEARCH")[-1].strip(": \n")
                step_sources = await self.retriever.retrieve(search_term, top_k=2)
                sources.extend(step_sources)
                context = "\n".join([s.text for s in step_sources])
                history.append(
                    f"Searched: {search_term}\nFound context length: {len(context)}"
                )
            else:
                break

        return await self._execute_vector(query)
