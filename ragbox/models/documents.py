"""
Document models for RAGBox.
Defines Pydantic models for various document types.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    CODE = "code"
    IMAGE = "image"
    STRUCTURED = "structured"
    TEXT = "text"
    UNKNOWN = "unknown"


class Document(BaseModel):
    """Base class for all documents."""

    id: str = Field(description="Unique content-addressed ID (SHA-256)")
    path: Path = Field(description="Original file path")
    content: str = Field(description="Extracted raw text content", default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_type: DocumentType = Field(default=DocumentType.UNKNOWN)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PDFDocument(Document):
    """PDF Document model with specific metadata."""

    doc_type: DocumentType = Field(default=DocumentType.PDF)
    tables: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted tables"
    )
    page_count: int = Field(default=0)


class CodeDocument(Document):
    """Source code document model."""

    doc_type: DocumentType = Field(default=DocumentType.CODE)
    language: str = Field(description="Programming language detected")
    ast_nodes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tree-sitter parsed AST"
    )


class ImageDocument(Document):
    """Image document model."""

    doc_type: DocumentType = Field(default=DocumentType.IMAGE)
    caption: str = Field(default="", description="Generated caption")
    ocr_text: str = Field(default="", description="Text extracted via OCR")


class StructuredDocument(Document):
    """Structured data document (JSON, CSV, XML)."""

    doc_type: DocumentType = Field(default=DocumentType.STRUCTURED)
    schema_inferred: Dict[str, Any] = Field(
        default_factory=dict, description="Inferred schema"
    )
    parsed_records: List[Dict[str, Any]] = Field(default_factory=list)
