"""
RAGBox FastAPI Server — HTTP API for RAGBox.

Install: pip install ragbox-core[server]
Usage: uvicorn ragbox.server:app --host 0.0.0.0 --port 8000
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the RAGBox server. "
        "Install with: pip install ragbox-core[server]"
    )

from ragbox import RAGBox

# Configuration
DOCUMENT_DIR = os.getenv("RAGBOX_DOCUMENT_DIR", "/data")

app = FastAPI(
    title="RAGBox API",
    description="Live, citation-first retrieval over local code and documents.",
    version="3.0.1",
)

# Global RAGBox instance (initialized on startup)
_ragbox: Optional[RAGBox] = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    cost_estimate: str = ""


class HealthResponse(BaseModel):
    status: str
    document_dir: str
    indexed: bool
    error: Optional[str] = None


@app.on_event("startup")
async def startup():
    global _ragbox
    doc_path = Path(DOCUMENT_DIR)
    if doc_path.exists():
        logger.info(f"Initializing RAGBox server with documents at {DOCUMENT_DIR}")
        _ragbox = RAGBox(doc_path)
        try:
            await _ragbox.aready()
        except Exception as exc:
            logger.error(f"RAGBox startup indexing failed: {exc}")
    else:
        logger.warning(
            f"Document directory {DOCUMENT_DIR} not found. "
            "Mount your docs with -v ./docs:/data"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    status = _ragbox.status if _ragbox else "no_documents"
    return HealthResponse(
        status=status,
        document_dir=DOCUMENT_DIR,
        indexed=bool(_ragbox and _ragbox.is_ready),
        error=_ragbox.build_error if _ragbox else None,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system and get a complete answer."""
    if not _ragbox:
        raise HTTPException(
            status_code=503,
            detail="RAGBox not initialized. Mount documents at /data.",
        )
    if _ragbox.status == "error":
        raise HTTPException(status_code=503, detail=_ragbox.build_error)

    try:
        answer = await _ragbox.aquery(request.question)
        cost = await _ragbox.aestimate_cost(query=request.question)
        return QueryResponse(answer=answer, cost_estimate=cost)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream the answer token-by-token using Server-Sent Events."""
    if not _ragbox:
        raise HTTPException(
            status_code=503,
            detail="RAGBox not initialized. Mount documents at /data.",
        )
    if _ragbox.status == "error":
        raise HTTPException(status_code=503, detail=_ragbox.build_error)

    async def generate():
        try:
            async for chunk in _ragbox.astream(request.question):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/stats")
async def stats():
    """Get cost and circuit breaker statistics."""
    if not _ragbox:
        return {"status": "not_initialized"}

    return {
        "status": _ragbox.status,
        "circuit_breaker": _ragbox.llm_client.circuit_breaker.get_stats(),
    }


@app.on_event("shutdown")
async def shutdown():
    if _ragbox is not None:
        await _ragbox.aclose()
