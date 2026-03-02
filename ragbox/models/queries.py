"""
Query and output models for RAGBox.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RAGStrategy(str, Enum):
    """Routing strategies for Agentic Orchestrator."""

    VECTOR = "vector"
    AGENTIC = "agentic"
    GRAPH = "graph"
    MULTI_QUERY = "multi_query"


class Source(BaseModel):
    """A documented source contributing to an answer."""

    document_id: str = Field(description="Source document ID")
    chunk_id: Optional[str] = Field(
        default=None, description="Specific chunk ID if applicable"
    )
    text: str = Field(description="Extract of the text used")
    score: float = Field(default=0.0, description="Relevance score")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Query(BaseModel):
    """An incoming query to the system."""

    text: str = Field(description="The user query text")
    mode: Optional[RAGStrategy] = Field(
        default=None, description="Forced strategy, or None for auto"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Answer(BaseModel):
    """Final output to the user."""

    query: str = Field(description="The original query")
    content: str = Field(description="The synthesized answer text")
    sources: List[Source] = Field(default_factory=list)
    strategy_used: RAGStrategy = Field(description="Strategy used to answer")
    execution_time_ms: float = Field(default=0.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)
