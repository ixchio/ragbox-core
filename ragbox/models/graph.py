"""
Graph models for Knowledge Graph components in RAGBox.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """An extracted named entity."""

    id: str = Field(description="Unique entity ID")
    name: str = Field(description="Name of the entity")
    entity_type: str = Field(
        description="Inferred type of the entity (e.g. PERSON, ORG)"
    )
    description: str = Field(default="", description="Description of the entity")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Relation(BaseModel):
    """A semantic relationship between two entities."""

    source_id: str = Field(description="ID of the source entity")
    target_id: str = Field(description="ID of the target entity")
    relation_type: str = Field(description="Type of relationship")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    context: str = Field(default="", description="Supporting text context")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Community(BaseModel):
    """A detected community of entities."""

    id: str = Field(description="Unique community ID")
    level: int = Field(description="Hierarchical level of the community")
    entity_ids: List[str] = Field(
        default_factory=list, description="Entities in this community"
    )
    summary: str = Field(
        default="", description="LLM-generated summary of the community"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GraphQueryResult(BaseModel):
    """Result of querying the knowledge graph."""

    relevant_entities: List[Entity] = Field(default_factory=list)
    relevant_relations: List[Relation] = Field(default_factory=list)
    relevant_communities: List[Community] = Field(default_factory=list)
    synthesized_context: str = Field(default="")
