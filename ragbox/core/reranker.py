import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger

class CrossEncoderReranker:
    """
    Second-stage scoring using a more powerful Cross-Encoder model.
    Stage 1: Vector Search (FAST, approximate) -> 100 candidates
    Stage 2: Cross-Encoder Rerank (SLOW, accurate) -> Top K candidates
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
        
    def _init_model(self):
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading Cross-Encoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name, max_length=512)
                self._initialized = True
            except ImportError:
                logger.warning("sentence_transformers required for local Cross-Encoder. Reranking disabled.")
                self._initialized = True # Mark true to avoid continuous import attempts
            except Exception as e:
                logger.error(f"Failed to load Cross-Encoder: {e}")
                self._initialized = True
                
    async def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate documents against the query.
        candidates should be a dict with at least 'content' and 'id'/'metadata'
        """
        # Lazy load to avoid blocking initialization
        if not self._initialized:
            await asyncio.to_thread(self._init_model)
            
        if not self._model or not candidates:
            # Fallback: Just return the original order truncated if model unavailable
            return candidates[:top_k]
            
        def _rerank_sync() -> List[Dict[str, Any]]:
            # Prepare pairs of (query, document) for the Cross-Encoder
            pairs = []
            valid_candidates = []
            
            for doc in candidates:
                content = doc.get("content", "")
                if content:
                    pairs.append([query, content])
                    valid_candidates.append(doc)
                    
            if not pairs:
                return []
                
            # Score pairs
            try:
                scores = self._model.predict(pairs)
                
                # Attach scores and sort
                for doc, score in zip(valid_candidates, scores):
                    doc["cross_encoder_score"] = float(score)
                    
                # Sort descending by cross encoder score
                reranked = sorted(valid_candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
                return reranked[:top_k]
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                return valid_candidates[:top_k]
                
        return await asyncio.to_thread(_rerank_sync)
