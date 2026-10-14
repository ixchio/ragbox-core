import os
import asyncio
import threading
from typing import List, Dict, Any
from loguru import logger


class CrossEncoderReranker:
    """
    Second-stage scoring using a more powerful Cross-Encoder model.
    Stage 1: Vector Search (FAST, approximate) -> 100 candidates
    Stage 2: Cross-Encoder Rerank (SLOW, accurate) -> Top K candidates
    """

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._init_lock = threading.Lock()

    def _init_model(self):
        with self._init_lock:
            if self._initialized:
                return
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info(f"Loading Cross-Encoder model: {self.model_name}")
                self._model = TextCrossEncoder(
                    model_name=self.model_name,
                    threads=max(1, (os.cpu_count() or 2) // 2),
                    providers=["CPUExecutionProvider"],
                )
            except ImportError:
                logger.warning(
                    "fastembed is required for local Cross-Encoder reranking. "
                    "Reranking disabled."
                )
            except Exception as e:
                logger.error(f"Failed to load Cross-Encoder: {e}")
            finally:
                self._initialized = True

    async def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate documents against the query.
        candidates should be a dict with at least 'content' and 'id'/'metadata'
        """
        # Lazy load to avoid blocking initialization
        if not self._initialized:
            await asyncio.to_thread(self._init_model)

        if not self._model or not candidates:
            # Preserve RRF order when the optional model is unavailable.
            fallback = sorted(candidates[:8], key=lambda item: item.get("rrf_rank", 0))
            for doc in fallback:
                doc["ce_rank"] = doc.get("rrf_rank", 0)
            return fallback[:top_k]

        def _rerank_sync() -> List[Dict[str, Any]]:
            # Prepare pairs of (query, document) for the Cross-Encoder
            documents = []
            valid_candidates = []

            for doc in candidates[:8]:
                content = doc.get("content", "")
                if content:
                    documents.append(content[:1200])
                    valid_candidates.append(doc)

            if not documents:
                return []

            # Score pairs
            try:
                scores = list(self._model.rerank(query, documents, batch_size=8))

                # Attach raw model scores, then convert them to scale-free ranks.
                for doc, score in zip(valid_candidates, scores):
                    doc["cross_encoder_score"] = float(score)

                reranked = sorted(
                    valid_candidates,
                    key=lambda item: (
                        -item["cross_encoder_score"],
                        item.get("rrf_rank", 0),
                    ),
                )
                for ce_rank, doc in enumerate(reranked):
                    doc["ce_rank"] = ce_rank
                return reranked[:top_k]

            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                return valid_candidates[:top_k]

        return await asyncio.to_thread(_rerank_sync)
