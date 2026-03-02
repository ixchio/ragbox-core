"""
Chunk models for RAGBox.
Defines chunking data structures.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """Base generic text chunk model."""

    id: str = Field(
        description="Unique chunk ID (SHA-256 of content + doc_id + position)"
    )
    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextChunk(Chunk):
    """Standard text chunk."""

    pass


class CodeChunk(Chunk):
    """Code chunk bounded by AST structures."""

    language: str = Field(description="Code language")
    function_name: Optional[str] = Field(default=None)
    class_name: Optional[str] = Field(default=None)


class TableChunk(Chunk):
    """Tabular data chunk."""

    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)


class ImageChunk(Chunk):
    """Image caption and OCR chunk."""

    caption: str = Field(default="")
    ocr_text: str = Field(default="")
