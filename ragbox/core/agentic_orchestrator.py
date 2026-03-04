"""
Layer 6: AGENTIC ORCHESTRATION
Query classification, dynamic routing, ReAct planning, and synthesis.
"""
import time
from typing import Any, Optional
from loguru import logger

from ragbox.models.queries import Answer, RAGStrategy
from ragbox.core.retrieval_fusion import RetrievalFusionEngine
from ragbox.utils.llm_clients import LLMClient


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
        """Classify query to route to correct strategy."""
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
        """Execute the end-to-end RAG pipeline."""
        start_time = time.time()

        strategy = force_strategy or await self._classify_query(query_text)
        logger.info(f"Executing query '{query_text}' with strategy {strategy.name}")

        if strategy == RAGStrategy.AGENTIC:
            answer = await self._execute_agentic(query_text)
        elif strategy == RAGStrategy.GRAPH:
            answer = await self._execute_graph(query_text)
        elif strategy == RAGStrategy.MULTI_QUERY:
            answer = await self._execute_multi_query(query_text)
        else:
            answer = await self._execute_vector(query_text)

        answer.execution_time_ms = (time.time() - start_time) * 1000
        return answer

    async def stream_execute(self, query_text: str):
        """Stream the answer token-by-token. Retrieves context first, then streams LLM generation."""
        strategy = await self._classify_query(query_text)
        logger.info(f"Streaming query '{query_text}' with strategy {strategy.name}")

        # Retrieve context (same as normal execute)
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

    async def _execute_vector(self, query: str) -> Answer:
        sources = await self.retriever.retrieve(query)
        context = "\n\n---\n\n".join([s.text for s in sources])

        prompt = f"Answer the query based ONLY on the following context.\n\nContext:\n{context}\n\nQuery: {query}"
        response = await self.llm.agenerate(
            prompt, system="You are an expert Q&A engine. Be concise and accurate."
        )

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

        # Deduplicate sources by text chunk ID or content to avoid noise
        seen = set()
        unique_sources = []
        for s in all_sources:
            # simple dedup
            if s.text not in seen:
                seen.add(s.text)
                unique_sources.append(s)

        # Limit to top N unique (since reranker will do the heavy lifting later in the pipeline anyway)
        unique_sources = unique_sources[:10]

        context = "\n\n---\n\n".join([s.text for s in unique_sources])
        prompt = f"Answer the query comprehensively based ONLY on the following context derived from multi-query expansion.\n\nContext:\n{context}\n\nQuery: {query}"
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

        prompt = f"Answer this query using the graph summaries and retrieved text provided below:\n\nContext:\n{context}\n\nQuery: {query}"
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
