"""
Layer 4: KNOWLEDGE GRAPH
Entity extraction, Leiden/Louvain community detection, real graph traversal,
and community summarization for cross-document reasoning.
"""

from typing import List, Tuple, Dict, Any, Set, Optional
import ast
import asyncio
import hashlib
import json
import re
import threading
from contextlib import contextmanager
from loguru import logger
import networkx as nx
import igraph as ig
import leidenalg
import community as community_louvain  # python-louvain

from ragbox.models.documents import Document, DocumentType
from ragbox.models.graph import Entity, Relation, Community, GraphQueryResult
from ragbox.utils.llm_clients import LLMClient


class OptimizedKnowledgeGraph:
    def __init__(
        self, optimize_threshold: int = 100, llm_client: Optional[LLMClient] = None
    ):
        self.graph = nx.Graph()
        self.llm = llm_client
        self.communities: Dict[int, Set[str]] = {}
        self.node_to_community: Dict[str, int] = {}
        self.community_summaries: Dict[int, str] = {}

        # Entity metadata — the key missing piece for real GraphRAG
        self.entity_descriptions: Dict[str, str] = {}
        self.entity_source_texts: Dict[str, List[str]] = {}
        self.relation_contexts: Dict[Tuple[str, str], List[str]] = {}

        # Incremental update tracking
        self.pending_updates = 0
        self.optimize_threshold = optimize_threshold
        self.last_entity_count = 0

        self.max_nodes_for_full_leiden = 50000
        self._lock = threading.RLock()
        self._document_entities: Dict[str, Set[str]] = {}
        self._document_entity_descriptions: Dict[str, Dict[str, str]] = {}
        self._document_entity_source_texts: Dict[str, Dict[str, str]] = {}
        self._document_relations: Dict[str, List[Dict[str, Any]]] = {}
        self._canonical_entities: Dict[str, str] = {}

    @contextmanager
    def mutation_lock(self):
        """Serialize graph reads with atomic document-revision commits."""
        with self._lock:
            yield

    @staticmethod
    def _edge_key(source: str, target: str) -> Tuple[str, str]:
        first, second = sorted((source, target))
        return first, second

    @staticmethod
    def _entity_key(entity: str) -> str:
        return " ".join(entity.split()).casefold()

    def _canonical_entity(self, entity: str) -> str:
        clean = " ".join(entity.split())
        key = self._entity_key(clean)
        existing = self._canonical_entities.get(key)
        if existing is not None and self.graph.has_node(existing):
            return existing
        self._canonical_entities[key] = clean
        return clean

    def add_document(
        self,
        doc_id: str,
        entities: List[str],
        relationships: List[Dict[str, Any]],
        entity_descriptions: Optional[Dict[str, str]] = None,
        entity_source_texts: Optional[Dict[str, str]] = None,
        *,
        optimize: bool = True,
    ) -> None:
        with self._lock:
            if doc_id in self._document_entities:
                self.remove_document(doc_id, optimize=False)

            entity_set = {
                self._canonical_entity(entity) for entity in entities if entity
            }
            canonical_descriptions = {
                self._canonical_entity(name): description
                for name, description in (entity_descriptions or {}).items()
                if name
            }
            canonical_sources = {
                self._canonical_entity(name): source_text
                for name, source_text in (entity_source_texts or {}).items()
                if name
            }
            clean_relations = []
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                if source and target:
                    source = self._canonical_entity(source)
                    target = self._canonical_entity(target)
                    entity_set.update((source, target))
                    clean_relations.append({**rel, "source": source, "target": target})

            self._document_entities[doc_id] = entity_set
            self._document_entity_descriptions[doc_id] = canonical_descriptions
            self._document_entity_source_texts[doc_id] = canonical_sources
            self._document_relations[doc_id] = clean_relations

            new_nodes = 0
            new_edges = 0
            for entity in entity_set:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, source_docs=set(), weight=1.0)
                    new_nodes += 1
                source_docs = self.graph.nodes[entity].setdefault("source_docs", set())
                source_docs.add(doc_id)
                self.graph.nodes[entity]["weight"] = float(len(source_docs))
                self._rebuild_entity_metadata(entity)

            for rel in clean_relations:
                source = rel["source"]
                target = rel["target"]
                rel_type = rel.get("type", "related_to")
                if not self.graph.has_edge(source, target):
                    self.graph.add_edge(
                        source,
                        target,
                        type=rel_type,
                        types=set(),
                        weight=1.0,
                        docs=set(),
                    )
                    new_edges += 1
                self.graph[source][target].setdefault("docs", set()).add(doc_id)
                self._rebuild_relation_metadata(source, target)

            self.pending_updates += new_nodes + new_edges
            ratio = self.pending_updates / max(1, len(self.graph.nodes))
            if optimize and (
                self.pending_updates > self.optimize_threshold or ratio > 0.1
            ):
                self._optimize_graph()

    def _rebuild_entity_metadata(self, entity: str) -> None:
        if not self.graph.has_node(entity):
            return
        descriptions = []
        sources = []
        for doc_id in self.graph.nodes[entity].get("source_docs", ()):
            description = self._document_entity_descriptions.get(doc_id, {}).get(entity)
            source_text = self._document_entity_source_texts.get(doc_id, {}).get(entity)
            if description:
                descriptions.append(description)
            if source_text:
                sources.append(source_text)
        if descriptions:
            self.entity_descriptions[entity] = descriptions[-1]
        else:
            self.entity_descriptions.pop(entity, None)
        if sources:
            self.entity_source_texts[entity] = sources
        else:
            self.entity_source_texts.pop(entity, None)

    def _rebuild_relation_metadata(self, source: str, target: str) -> None:
        if not self.graph.has_edge(source, target):
            return
        edge_key = self._edge_key(source, target)
        types: Set[str] = set()
        contexts = []
        edge = self.graph[source][target]
        docs: Set[str] = set(edge.get("docs", set()))
        for doc_id in docs:
            for rel in self._document_relations.get(doc_id, ()):
                if self._edge_key(rel["source"], rel["target"]) != edge_key:
                    continue
                types.add(rel.get("type", "related_to"))
                context = rel.get("context", "")
                if context:
                    contexts.append(context)

        edge["docs"] = docs
        edge["types"] = types
        edge["type"] = sorted(types)[0] if types else "related_to"
        edge["weight"] = float(max(1, len(docs)))
        if contexts:
            self.relation_contexts[edge_key] = contexts
        else:
            self.relation_contexts.pop(edge_key, None)

    def remove_document(self, doc_id: str, *, optimize: bool = True) -> bool:
        """Remove every node, edge, context and description owned by a revision."""
        if not doc_id:
            return False
        with self._lock:
            entities = set(self._document_entities.pop(doc_id, set()))
            relations = self._document_relations.pop(doc_id, [])
            self._document_entity_descriptions.pop(doc_id, None)
            self._document_entity_source_texts.pop(doc_id, None)

            # Support graphs created through older in-memory code paths.
            if not entities:
                entities = {
                    node
                    for node, data in self.graph.nodes(data=True)
                    if doc_id in data.get("source_docs", set())
                }

            affected_edges = {
                self._edge_key(rel["source"], rel["target"])
                for rel in relations
                if rel.get("source") and rel.get("target")
            }
            for source, target, data in list(self.graph.edges(data=True)):
                if doc_id in data.get("docs", set()):
                    affected_edges.add(self._edge_key(source, target))

            for source, target in affected_edges:
                if not self.graph.has_edge(source, target):
                    continue
                docs = self.graph[source][target].setdefault("docs", set())
                docs.discard(doc_id)
                if not docs:
                    self.graph.remove_edge(source, target)
                    self.relation_contexts.pop((source, target), None)
                else:
                    self._rebuild_relation_metadata(source, target)

            for entity in entities:
                if not self.graph.has_node(entity):
                    continue
                source_docs = self.graph.nodes[entity].setdefault("source_docs", set())
                source_docs.discard(doc_id)
                if not source_docs:
                    self.graph.remove_node(entity)
                    key = self._entity_key(entity)
                    if self._canonical_entities.get(key) == entity:
                        self._canonical_entities.pop(key, None)
                    self.entity_descriptions.pop(entity, None)
                    self.entity_source_texts.pop(entity, None)
                    self.node_to_community.pop(entity, None)
                else:
                    self.graph.nodes[entity]["weight"] = float(len(source_docs))
                    self._rebuild_entity_metadata(entity)

            self.pending_updates += max(1, len(entities) + len(affected_edges))
            self.community_summaries.clear()
            if not self.graph.nodes:
                self.communities.clear()
                self.node_to_community.clear()
                self.pending_updates = 0
            elif optimize:
                self._optimize_graph()
            return bool(entities or affected_edges)

    def snapshot_document(self, doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Copy one revision's graph contribution for transactional rollback."""
        if not doc_id:
            return None
        with self._lock:
            if doc_id not in self._document_entities:
                return None
            return {
                "doc_id": doc_id,
                "entities": sorted(self._document_entities[doc_id]),
                "relationships": [
                    dict(relation)
                    for relation in self._document_relations.get(doc_id, [])
                ],
                "entity_descriptions": dict(
                    self._document_entity_descriptions.get(doc_id, {})
                ),
                "entity_source_texts": dict(
                    self._document_entity_source_texts.get(doc_id, {})
                ),
            }

    def _clear_graph_state(self) -> None:
        self.graph.clear()
        self.communities.clear()
        self.node_to_community.clear()
        self.community_summaries.clear()
        self.entity_descriptions.clear()
        self.entity_source_texts.clear()
        self.relation_contexts.clear()
        self._document_entities.clear()
        self._document_entity_descriptions.clear()
        self._document_entity_source_texts.clear()
        self._document_relations.clear()
        self._canonical_entities.clear()
        self.pending_updates = 0
        self.last_entity_count = 0

    def _community_signature(self, members: Set[str]) -> str:
        """Fingerprint exactly the graph content used to produce a summary."""
        member_set = set(members)
        edges = []
        for source, target, data in self.graph.edges(data=True):
            if source not in member_set or target not in member_set:
                continue
            first, second = self._edge_key(source, target)
            types = data.get("types", set())
            if not types:
                types = {data.get("type", "related_to")}
            edges.append((first, second, sorted(types)))
        payload = {
            "members": sorted(member_set),
            "descriptions": {
                member: self.entity_descriptions.get(member, "")
                for member in sorted(member_set)
            },
            "edges": sorted(edges),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def export_community_summaries(self) -> Dict[str, str]:
        """Export summaries keyed by stable community-content fingerprints."""
        with self._lock:
            return {
                self._community_signature(members): summary
                for comm_idx, members in self.communities.items()
                if (summary := self.community_summaries.get(comm_idx))
            }

    def _restore_community_summaries(self, cached: Dict[str, str]) -> None:
        self.community_summaries = {
            comm_idx: cached[signature]
            for comm_idx, members in self.communities.items()
            if (signature := self._community_signature(members)) in cached
        }

    def restore_snapshots(
        self,
        snapshots: List[Dict[str, Any]],
        cached_summaries: Optional[Dict[str, str]] = None,
    ) -> None:
        """Rebuild the complete in-memory graph from persisted active revisions."""
        required = {
            "doc_id",
            "entities",
            "relationships",
            "entity_descriptions",
            "entity_source_texts",
        }
        seen_doc_ids = set()
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or not required.issubset(snapshot):
                raise ValueError("Invalid persisted graph snapshot")
            doc_id = snapshot["doc_id"]
            if not isinstance(doc_id, str) or not doc_id or doc_id in seen_doc_ids:
                raise ValueError(f"Invalid or duplicate graph revision: {doc_id!r}")
            if not isinstance(snapshot["entities"], list) or not isinstance(
                snapshot["relationships"], list
            ):
                raise ValueError(f"Invalid graph entities/relationships for {doc_id}")
            if not isinstance(snapshot["entity_descriptions"], dict) or not isinstance(
                snapshot["entity_source_texts"], dict
            ):
                raise ValueError(f"Invalid graph metadata for {doc_id}")
            seen_doc_ids.add(doc_id)

        with self._lock:
            self._clear_graph_state()
            try:
                for snapshot in snapshots:
                    self.add_document(
                        doc_id=snapshot["doc_id"],
                        entities=snapshot["entities"],
                        relationships=snapshot["relationships"],
                        entity_descriptions=snapshot["entity_descriptions"],
                        entity_source_texts=snapshot["entity_source_texts"],
                        optimize=False,
                    )
                if self.graph.nodes:
                    self._optimize_graph()
                    self._restore_community_summaries(cached_summaries or {})
            except Exception:
                self._clear_graph_state()
                raise

    def rollback_document_replacement(
        self, new_doc_id: str, old_snapshot: Optional[Dict[str, Any]]
    ) -> None:
        """Restore graph state after a revision activation fails."""
        with self._lock:
            self.remove_document(new_doc_id, optimize=False)
            if old_snapshot:
                self.add_document(
                    doc_id=old_snapshot["doc_id"],
                    entities=old_snapshot["entities"],
                    relationships=old_snapshot["relationships"],
                    entity_descriptions=old_snapshot["entity_descriptions"],
                    entity_source_texts=old_snapshot["entity_source_texts"],
                    optimize=False,
                )
            self.community_summaries.clear()
            if self.graph.nodes:
                self._optimize_graph()

    # ── Community Detection ────────────────────────────────────────────────

    def _optimize_graph(self) -> None:
        if len(self.graph.nodes) == 0:
            return

        logger.info(
            f"Optimizing Knowledge Graph "
            f"(Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)})"
        )
        try:
            if len(self.graph.nodes) > self.max_nodes_for_full_leiden:
                self._detect_communities_louvain()
            else:
                self._detect_communities_leiden()
            self.pending_updates = 0
            self.last_entity_count = len(self.graph.nodes)
        except Exception as e:
            logger.error(f"Failed to optimize graph: {e}")
            self._detect_communities_fallback()

    def _detect_communities_leiden(self) -> None:
        logger.debug("Running Leiden algorithm for community detection")
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        reverse_mapping = {i: node for node, i in node_mapping.items()}
        edges = [(node_mapping[u], node_mapping[v]) for u, v in self.graph.edges()]
        weights = [
            data.get("weight", 1.0) for _, _, data in self.graph.edges(data=True)
        ]
        ig_graph = ig.Graph(n=len(self.graph.nodes()), edges=edges)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights=weights,
            n_iterations=2,
        )
        self.communities.clear()
        self.node_to_community.clear()
        for idx, community_idx in enumerate(partition.membership):
            node_name = reverse_mapping[idx]
            self.node_to_community[node_name] = community_idx
            self.communities.setdefault(community_idx, set()).add(node_name)
        logger.info(f"Detected {len(self.communities)} communities via Leiden")

    def _detect_communities_louvain(self) -> None:
        logger.debug("Running Louvain algorithm for large graph community detection")
        partition = community_louvain.best_partition(self.graph, weight="weight")
        self.communities.clear()
        self.node_to_community.clear()
        for node, community_idx in partition.items():
            self.node_to_community[node] = community_idx
            self.communities.setdefault(community_idx, set()).add(node)
        logger.info(f"Detected {len(self.communities)} communities via Louvain")

    def _detect_communities_fallback(self) -> None:
        logger.warning("Using connected components fallback for communities")
        self.communities.clear()
        self.node_to_community.clear()
        for idx, component in enumerate(nx.connected_components(self.graph)):
            self.communities[idx] = set(component)
            for node in component:
                self.node_to_community[node] = idx

    # ── Entity Lookup ──────────────────────────────────────────────────────

    def get_community_for_entity(self, entity: str) -> Optional[int]:
        return self.node_to_community.get(entity)

    def get_related_entities(self, entity: str, max_distance: int = 2) -> List[str]:
        if not self.graph.has_node(entity):
            return []
        if self.pending_updates > self.optimize_threshold:
            self._optimize_graph()

        related = set()
        neighbors = set(self.graph.neighbors(entity))
        related.update(neighbors)

        comm_idx = self.get_community_for_entity(entity)
        if comm_idx is not None and comm_idx in self.communities:
            comm_members = list(self.communities[comm_idx])
            related.update(comm_members[:15])

        if max_distance > 1:
            for neighbor in list(neighbors):
                related.update(self.graph.neighbors(neighbor))

        related.discard(entity)
        return list(related)

    def set_community_summary(self, comm_idx: int, summary: str) -> None:
        self.community_summaries[comm_idx] = summary

    def get_community_summary(self, comm_idx: int) -> Optional[str]:
        """Get cached community summary"""
        return self.community_summaries.get(comm_idx)

    def extract_subgraph(self, entities: List[str]) -> Dict[str, Any]:
        """Extract a relevant subgraph for context injection."""
        valid_entities = [e for e in entities if self.graph.has_node(e)]
        if not valid_entities:
            return {"nodes": [], "edges": []}

        subgraph = self.graph.subgraph(valid_entities).copy()
        nodes = []
        for n in subgraph.nodes():
            node_data = dict(subgraph.nodes[n])
            if "source_docs" in node_data:
                node_data["source_docs"] = list(node_data["source_docs"])
            nodes.append({"id": n, **node_data})

        edges = []
        for u, v, data in subgraph.edges(data=True):
            edge_data = dict(data)
            if "docs" in edge_data:
                edge_data["docs"] = list(edge_data["docs"])
            if "types" in edge_data:
                edge_data["types"] = list(edge_data["types"])
            edges.append({"source": u, "target": v, **edge_data})

        return {"nodes": nodes, "edges": edges}

    # ── Entity Matching — the key to real GraphRAG ─────────────────────────

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def _match_query_to_entities(self, query: str) -> List[Tuple[str, float]]:
        """
        Match query text to graph entities using multi-signal scoring:
        1. Exact substring match (highest weight)
        2. Token overlap (medium weight)
        3. Case-insensitive containment
        Returns list of (entity_name, score) sorted by score descending.
        """
        if not self.graph.nodes:
            return []

        query_lower = query.casefold()
        query_tokens = {
            token
            for token in re.findall(r"\w+", query_lower)
            if len(token) > 2 and token not in self._STOPWORDS
        }
        if not query_tokens:
            return []
        scored: Dict[str, float] = {}

        for entity in self.graph.nodes():
            entity_lower = entity.casefold()
            entity_tokens = {
                token
                for token in re.findall(r"\w+", entity_lower)
                if token not in self._STOPWORDS
            }
            score = 0.0

            # Exact substring match (entity appears literally in query)
            if len(entity_lower) > 3 and entity_lower in query_lower:
                score += 1.0
            # Entity name contains a query word
            elif query_tokens & entity_tokens:
                overlap = len(query_tokens & entity_tokens)
                score += 0.6 * overlap / max(len(entity_tokens), 1)

            # Check description overlap
            desc = self.entity_descriptions.get(entity, "").lower()
            if desc:
                desc_tokens = set(re.findall(r"\w+", desc))
                desc_overlap = len(query_tokens & desc_tokens)
                if desc_overlap > 0:
                    score += 0.25 * desc_overlap / max(len(query_tokens), 1)

            if score > 0:
                # Boost by graph centrality (well-connected entities are more important)
                degree = self.graph.degree(entity)
                score *= 1.0 + 0.05 * min(degree, 10)
                scored[entity] = score

        results = sorted(
            scored.items(),
            key=lambda item: (-item[1], item[0].casefold(), item[0]),
        )
        return results[:15]

    def _igraph_view(
        self,
    ) -> Tuple[ig.Graph, Dict[str, int], Dict[int, str], List[float]]:
        """Build an igraph view with stable node indexes and edge weights."""
        nodes = list(self.graph.nodes())
        node_to_idx = {name: idx for idx, name in enumerate(nodes)}
        idx_to_node = {idx: name for name, idx in node_to_idx.items()}
        edges = [
            (node_to_idx[source], node_to_idx[target])
            for source, target in self.graph.edges()
        ]
        edge_weights = [
            float(data.get("weight", 1.0)) for _, _, data in self.graph.edges(data=True)
        ]
        return (
            ig.Graph(n=len(nodes), edges=edges, directed=False),
            node_to_idx,
            idx_to_node,
            edge_weights,
        )

    def _expand_with_personalized_pagerank(
        self, matched: List[Tuple[str, float]], limit: int = 12
    ) -> List[str]:
        """Keep matched seeds first, then rank connected nodes by weighted PPR."""
        seeds = [name for name, _ in matched[:10] if self.graph.has_node(name)]
        if not seeds:
            return []

        match_scores = dict(matched)
        ig_graph, node_to_idx, idx_to_node, edge_weights = self._igraph_view()
        if ig_graph.ecount() == 0:
            return [name for name, _ in matched if name in node_to_idx][:limit]

        reset = [0.0] * ig_graph.vcount()
        for name, score in matched[:10]:
            node_idx = node_to_idx.get(name)
            if node_idx is not None:
                reset[node_idx] = float(score)

        if sum(reset) <= 0:
            return seeds[:limit]

        try:
            ppr_scores = ig_graph.personalized_pagerank(
                damping=0.85,
                reset=reset,
                weights=edge_weights,
            )
            seed_set = set(seeds)
            connected = [
                idx_to_node[idx]
                for idx, score in enumerate(ppr_scores)
                if score > 0.0 and idx_to_node[idx] not in seed_set
            ]
            connected.sort(
                key=lambda name: (
                    -float(ppr_scores[node_to_idx[name]]),
                    -float(match_scores.get(name, 0.0)),
                    name.casefold(),
                    name,
                )
            )
            return (seeds + connected)[:limit]
        except Exception as exc:
            logger.warning(f"Personalized PageRank failed; using traversal: {exc}")
            related = set()
            for seed in seeds:
                related.update(self.get_related_entities(seed, max_distance=2))
            related.difference_update(seeds)
            ordered_related = sorted(
                related,
                key=lambda name: (
                    -float(match_scores.get(name, 0.0)),
                    -int(self.graph.degree(name)) if self.graph.has_node(name) else 0,
                    name.casefold(),
                    name,
                ),
            )
            return (seeds + ordered_related)[:limit]

    def _collect_entity_context(self, entities: List[str]) -> str:
        """
        Collect rich textual context for a set of entities from the graph.
        Gathers: entity descriptions, source texts, relationship descriptions,
        and community summaries.
        """
        parts = []

        for entity in entities:
            entity_parts = [f"Entity: {entity}"]

            desc = self.entity_descriptions.get(entity)
            if desc:
                entity_parts.append(f"  Description: {desc}")

            # Source texts — actual document content mentioning this entity
            source_texts = self.entity_source_texts.get(entity, [])
            for text in source_texts[:2]:  # cap to avoid prompt explosion
                entity_parts.append(f"  Source: {text[:500]}")

            # Relationships
            if self.graph.has_node(entity):
                for neighbor in list(self.graph.neighbors(entity))[:10]:
                    edge_data = self.graph[entity][neighbor]
                    rel_type = edge_data.get("type", "related_to")
                    rel_types = edge_data.get("types", set())
                    type_str = ", ".join(rel_types) if rel_types else rel_type
                    entity_parts.append(f"  → {type_str} → {neighbor}")

                    # Relationship context text
                    contexts = self.relation_contexts.get(
                        self._edge_key(entity, neighbor), []
                    )
                    for ctx in contexts[:1]:
                        entity_parts.append(f"    Context: {ctx[:300]}")

            parts.append("\n".join(entity_parts))

        # Community summaries for touched communities
        touched_communities = set()
        for entity in entities:
            comm = self.get_community_for_entity(entity)
            if comm is not None:
                touched_communities.add(comm)

        for comm_idx in list(touched_communities)[:3]:
            summary = self.community_summaries.get(comm_idx)
            if summary:
                members = list(self.communities.get(comm_idx, set()))[:10]
                parts.append(
                    f"\nCommunity (entities: {', '.join(members)}):\n{summary}"
                )

        return "\n\n".join(parts)

    # ── Graph Query — the core of GraphRAG ────────────────────────────────

    async def query(self, query: str, mode: str = "hybrid") -> GraphQueryResult:
        # The body is synchronous, so holding this lock never blocks across an
        # await. Revision commits use the same lock and become atomic to reads.
        with self._lock:
            return self._query_locked(query, mode)

    def _query_locked(self, query: str, mode: str = "hybrid") -> GraphQueryResult:
        """
        Search the knowledge graph with real entity matching and traversal.

        1. Match query terms to graph entities (fuzzy + token overlap)
        2. Expand matched entities via graph traversal (neighbors + community)
        3. Collect entity descriptions, source texts, and relationship context
        4. Return structured result with synthesized context
        """
        if not self.graph.nodes:
            return GraphQueryResult(
                relevant_entities=[],
                relevant_relations=[],
                relevant_communities=[],
                synthesized_context="",
            )

        # Step 1: Match query to entities
        matched = self._match_query_to_entities(query)
        if not matched:
            return GraphQueryResult(
                relevant_entities=[],
                relevant_relations=[],
                relevant_communities=[],
                synthesized_context="",
            )

        seed_entities = [name for name, _ in matched[:10]]
        logger.info(
            f"Graph query matched {len(seed_entities)} seed entities: "
            f"{seed_entities[:5]}"
        )

        # Step 2: Expand through the C-backed graph using personalized PageRank.
        expanded_list = self._expand_with_personalized_pagerank(matched, limit=12)
        expanded = set(expanded_list)

        # Step 3: Collect rich context
        synthesized = self._collect_entity_context(expanded_list)

        # Step 4: Build structured result
        result_entities = []
        for name in expanded_list:
            desc = self.entity_descriptions.get(name, "")
            result_entities.append(
                Entity(
                    id=name,
                    name=name,
                    entity_type="EXTRACTED",
                    description=desc,
                )
            )

        result_relations = []
        seen_edges: Set[Tuple[str, str]] = set()
        for entity in expanded_list:
            if not self.graph.has_node(entity):
                continue
            for neighbor in self.graph.neighbors(entity):
                if neighbor in expanded and (entity, neighbor) not in seen_edges:
                    seen_edges.add((entity, neighbor))
                    seen_edges.add((neighbor, entity))
                    edge = self.graph[entity][neighbor]
                    result_relations.append(
                        Relation(
                            source_id=entity,
                            target_id=neighbor,
                            relation_type=edge.get("type", "related_to"),
                            context="; ".join(
                                self.relation_contexts.get(
                                    self._edge_key(entity, neighbor), []
                                )[:2]
                            ),
                        )
                    )

        touched_communities = set()
        for entity in expanded_list:
            comm = self.get_community_for_entity(entity)
            if comm is not None:
                touched_communities.add(comm)

        result_communities = []
        for comm_idx in list(touched_communities)[:5]:
            summary = self.community_summaries.get(comm_idx, "")
            members = list(self.communities.get(comm_idx, set()))
            result_communities.append(
                Community(
                    id=f"community_{comm_idx}",
                    level=1,
                    entity_ids=members,
                    summary=summary,
                )
            )

        return GraphQueryResult(
            relevant_entities=result_entities,
            relevant_relations=result_relations,
            relevant_communities=result_communities,
            synthesized_context=synthesized,
        )

    # ── Build Pipeline ────────────────────────────────────────────────────

    async def prepare_document(
        self, document: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Perform expensive extraction without mutating the active graph."""
        return await self._extract_single_doc(document)

    def commit_prepared_document(
        self,
        document: Document,
        prepared: Tuple[List[Dict[str, str]], List[Dict[str, str]]],
        *,
        replace_doc_id: Optional[str] = None,
        optimize: bool = True,
    ) -> None:
        """Atomically replace one document's complete graph contribution."""
        entities_data, relations_data = prepared
        entity_names = [
            entity["name"] for entity in entities_data if entity.get("name")
        ]
        entity_descs = {
            entity["name"]: entity.get("description", "")
            for entity in entities_data
            if entity.get("name")
        }
        doc_snippet = document.content[:2000]
        entity_sources = {name: doc_snippet for name in entity_names}
        rels = [
            {
                "source": relation["source"],
                "target": relation["target"],
                "type": relation.get("type", "RELATED_TO"),
                "context": relation.get("context", ""),
            }
            for relation in relations_data
            if relation.get("source") and relation.get("target")
        ]

        with self._lock:
            if replace_doc_id and replace_doc_id != document.id:
                self.remove_document(replace_doc_id, optimize=False)
            self.community_summaries.clear()
            self.add_document(
                doc_id=document.id,
                entities=entity_names,
                relationships=rels,
                entity_descriptions=entity_descs,
                entity_source_texts=entity_sources,
                optimize=False,
            )
            if optimize and self.graph.nodes:
                self._optimize_graph()

    async def refresh_community_summaries(self) -> None:
        """Refresh summaries after a committed graph mutation."""
        if not self.graph.nodes:
            self.community_summaries.clear()
            return
        # Snapshot community membership while mutations are excluded; LLM calls
        # happen after the commit and are best-effort derived data.
        with self._lock:
            cached = self.export_community_summaries()
            self._optimize_graph()
            self._restore_community_summaries(cached)
        await self._generate_community_summaries()

    async def build_from_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logger.info(
            f"Extracting entities and relations for {len(documents)} documents."
        )

        semaphore = asyncio.Semaphore(4)

        async def _prepare(doc: Document):
            async with semaphore:
                return doc, await self.prepare_document(doc)

        prepared_documents = await asyncio.gather(*(_prepare(doc) for doc in documents))

        with self._lock:
            for doc, prepared in prepared_documents:
                self.commit_prepared_document(doc, prepared, optimize=False)
            if self.graph.nodes:
                self._optimize_graph()

        # Community detection + summarization
        logger.info("Running community detection and summarization.")
        await self._generate_community_summaries()
        logger.info(
            f"Knowledge Graph build complete. "
            f"Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}, "
            f"Communities: {len(self.communities)}"
        )

    async def _extract_single_doc(
        self, doc: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Extract entities and relationships from a single document."""
        import json
        import hashlib

        code_suffixes = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
        }
        if (
            doc.doc_type == DocumentType.CODE
            or doc.path.suffix.lower() in code_suffixes
        ):
            return self._extract_entities_regex(doc)

        llm = self.llm
        if llm is None:
            logger.warning("No LLM client — using regex-based entity extraction.")
            return self._extract_entities_regex(doc)

        cache_dir = doc.path.parent / ".ragbox_state" / "graph_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        doc_hash = hashlib.md5(doc.content.encode()).hexdigest()
        cache_file = cache_dir / f"{doc.id}_{doc_hash}.json"

        if cache_file.exists():
            logger.debug(f"Loading cached graph extraction for {doc.id}")
            try:
                res = json.loads(cache_file.read_text())
                return res.get("entities", []), res.get("relationships", [])
            except Exception:
                pass

        schema = {
            "entities": [
                {
                    "name": "string",
                    "type": "PERSON|ORG|CONCEPT|METRIC|EVENT|POLICY|TEAM|ROLE|PRODUCT",
                    "description": "string",
                }
            ],
            "relationships": [
                {
                    "source": "entity name",
                    "target": "entity name",
                    "type": "string",
                    "context": "brief description of relationship",
                }
            ],
        }

        snippet = doc.content[:4000]
        prompt = (
            f"Extract ALL important entities (people, organizations, teams, roles, "
            f"policies, metrics, events, products, concepts) and their relationships "
            f"from this text. Be thorough — capture every named entity and how they "
            f"connect.\n\nText:\n{snippet}"
        )

        res = {}
        for attempt in range(3):
            try:
                res = await llm.agenerate_structured(
                    prompt,
                    schema,
                    system=(
                        "You are an expert knowledge graph builder. Extract entities "
                        "and relationships precisely. Use the EXACT entity names as "
                        "they appear in the text. Every relationship must reference "
                        "entities from your entities list."
                    ),
                )
                if res and (res.get("entities") or res.get("relationships")):
                    try:
                        cache_file.write_text(json.dumps(res))
                    except Exception as e:
                        logger.warning(f"Failed to write cache for {doc.id}: {e}")
                    break

                logger.warning(
                    f"Empty extraction on attempt {attempt + 1} for {doc.id}"
                )
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Extraction failed attempt {attempt + 1}: {e}")
                if "429" in str(e):
                    await asyncio.sleep(10)

        if not res:
            logger.warning(f"LLM extraction failed for {doc.id}, using regex fallback")
            return self._extract_entities_regex(doc)

        return res.get("entities", []), res.get("relationships", [])

    @staticmethod
    def _dotted_ast_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = OptimizedKnowledgeGraph._dotted_ast_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def _extract_python_ast(
        self, doc: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Build a deterministic Python symbol graph without any LLM calls."""
        tree = ast.parse(doc.content, filename=str(doc.path))
        entities_by_name: Dict[str, Dict[str, str]] = {}
        relationships: List[Dict[str, str]] = []
        seen_relationships: Set[Tuple[str, str, str]] = set()

        def add_entity(name: str, entity_type: str, description: str) -> None:
            clean_name = name.strip()
            if clean_name and clean_name not in entities_by_name:
                entities_by_name[clean_name] = {
                    "name": clean_name,
                    "type": entity_type,
                    "description": description,
                }

        def add_relationship(source: str, target: str, rel_type: str) -> None:
            key = (source, target, rel_type)
            if source and target and key not in seen_relationships:
                seen_relationships.add(key)
                relationships.append(
                    {
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "context": f"{source} {rel_type} {target} in {doc.path.name}",
                    }
                )

        parent_by_node: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_by_node[child] = parent

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_entity(
                        alias.name,
                        "MODULE",
                        f"Imported module in {doc.path.name}",
                    )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or "." * node.level
                if module_name:
                    add_entity(
                        module_name,
                        "MODULE",
                        f"Imported module in {doc.path.name}",
                    )

        symbol_nodes: List[
            Tuple[
                int,
                int,
                ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
                str,
                Optional[str],
            ]
        ] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbol_nodes.append(
                    (node.lineno, node.col_offset, node, node.name, None)
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                search_parent: Optional[ast.AST] = parent_by_node.get(node)
                owner: Optional[str] = None
                while search_parent is not None:
                    if isinstance(search_parent, ast.ClassDef):
                        owner = search_parent.name
                        break
                    search_parent = parent_by_node.get(search_parent)
                symbol_name = f"{owner}.{node.name}" if owner else node.name
                symbol_nodes.append(
                    (node.lineno, node.col_offset, node, symbol_name, owner)
                )

        symbol_nodes.sort(key=lambda item: (item[0], item[1], item[3]))
        function_nodes: List[
            Tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, Optional[str]]
        ] = []
        for _, _, node, symbol_name, owner in symbol_nodes:
            if isinstance(node, ast.ClassDef):
                add_entity(
                    symbol_name,
                    "CLASS",
                    f"Python class defined in {doc.path.name}:{node.lineno}",
                )
                for base in node.bases:
                    base_name = self._dotted_ast_name(base)
                    if base_name:
                        add_entity(
                            base_name,
                            "CLASS",
                            f"Base class referenced in {doc.path.name}",
                        )
                        add_relationship(symbol_name, base_name, "inherits_from")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_entity(
                    symbol_name,
                    "FUNCTION",
                    f"Python function defined in {doc.path.name}:{node.lineno}",
                )
                function_nodes.append((node, symbol_name, owner))
                if owner:
                    add_relationship(owner, symbol_name, "defines_method")

        symbols_by_simple_name: Dict[str, List[str]] = {}
        for entity_name, entity in entities_by_name.items():
            if entity["type"] not in {"CLASS", "FUNCTION"}:
                continue
            symbols_by_simple_name.setdefault(
                entity_name.rsplit(".", 1)[-1], []
            ).append(entity_name)

        def resolve_call(call: ast.Call, owner: Optional[str]) -> Optional[str]:
            called_name = self._dotted_ast_name(call.func)
            if not called_name:
                return None
            if called_name.startswith("self.") and owner:
                owned_method = f"{owner}.{called_name.rsplit('.', 1)[-1]}"
                if owned_method in entities_by_name:
                    return owned_method
            if called_name in entities_by_name:
                return called_name
            candidates = symbols_by_simple_name.get(called_name.rsplit(".", 1)[-1], [])
            return candidates[0] if len(candidates) == 1 else None

        class CallVisitor(ast.NodeVisitor):
            def __init__(self, root: ast.AST) -> None:
                self.root = root
                self.calls: List[ast.Call] = []

            def visit_Call(self, node: ast.Call) -> None:
                self.calls.append(node)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                if node is self.root:
                    self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                if node is self.root:
                    self.generic_visit(node)

        for node, caller, owner in function_nodes:
            visitor = CallVisitor(node)
            visitor.visit(node)
            for call in visitor.calls:
                target = resolve_call(call, owner)
                if target and target != caller:
                    add_relationship(caller, target, "calls")

        return list(entities_by_name.values()), relationships

    def _extract_code_regex(
        self, doc: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Extract common non-Python code symbols deterministically."""
        text = doc.content[:100_000]
        suffix = doc.path.suffix.lower()
        entities: List[Dict[str, str]] = []
        seen: Set[str] = set()

        def add_matches(pattern: str, entity_type: str) -> None:
            for match in re.finditer(pattern, text, re.MULTILINE):
                name = match.group(1).strip()
                if name and name not in seen:
                    seen.add(name)
                    entities.append(
                        {
                            "name": name,
                            "type": entity_type,
                            "description": f"Code symbol from {doc.path.name}",
                        }
                    )

        if suffix in {".js", ".ts", ".tsx"}:
            add_matches(r"\b(?:class|interface|enum)\s+([A-Za-z_$][\w$]*)", "CLASS")
            add_matches(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", "FUNCTION")
            add_matches(
                r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
                "FUNCTION",
            )
            add_matches(
                r"(?:import\s+.*?\s+from\s+|require\s*\(\s*)['\"]([^'\"]+)['\"]",
                "MODULE",
            )
        elif suffix == ".go":
            add_matches(r"\btype\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b", "CLASS")
            add_matches(r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(", "FUNCTION")
            add_matches(r"^\s*import\s+(?:\w+\s+)?[\"`]([^\"`]+)[\"`]", "MODULE")
        elif suffix == ".rs":
            add_matches(r"\b(?:struct|trait|enum)\s+([A-Za-z_]\w*)", "CLASS")
            add_matches(r"\bfn\s+([A-Za-z_]\w*)\s*\(", "FUNCTION")
            add_matches(r"^\s*use\s+([^;]+);", "MODULE")
        elif suffix == ".java":
            add_matches(r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)", "CLASS")
            add_matches(
                r"\b(?:public|protected|private|static|final|native|synchronized|abstract|default|\s)+[\w<>\[\],.?]+\s+([A-Za-z_]\w*)\s*\(",
                "FUNCTION",
            )
            add_matches(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", "MODULE")
        elif suffix in {".c", ".cpp"}:
            add_matches(r"\b(?:class|struct|enum)\s+([A-Za-z_]\w*)", "CLASS")
            add_matches(
                r"^\s*(?:[A-Za-z_]\w*[\s:*&<>]+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
                "FUNCTION",
            )
            add_matches(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]", "MODULE")

        relationships: List[Dict[str, str]] = []
        entity_names = [entity["name"] for entity in entities]
        for index, source in enumerate(entity_names[:10]):
            for target in entity_names[index + 1 : index + 4]:
                relationships.append(
                    {
                        "source": source,
                        "target": target,
                        "type": "co_occurs_with",
                        "context": f"Both defined in {doc.path.name}",
                    }
                )
        return entities, relationships

    def _extract_entities_regex(
        self, doc: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Regex-based entity extraction fallback. Finds capitalized proper nouns
        and common named entity patterns. No LLM needed.
        """
        suffix = doc.path.suffix.lower()
        if suffix == ".py":
            try:
                return self._extract_python_ast(doc)
            except SyntaxError as exc:
                logger.warning(f"Python AST parsing failed for {doc.path}: {exc}")
        elif suffix in {".js", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp"}:
            return self._extract_code_regex(doc)

        text = doc.content[:5000]

        # Find capitalized multi-word names (e.g. "Maria Santos", "VP of Engineering")
        name_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s+(?:of|the|and|for|in)\s+)?(?:[A-Z][a-z]+)+)\b"
        )
        names = name_pattern.findall(text)

        # Find role/title patterns
        role_pattern = re.compile(
            r"\b((?:VP|CEO|CTO|CFO|COO|Director|Manager|Lead|Head|Chief)\s+(?:of\s+)?[\w\s]{2,30})\b"
        )
        roles = role_pattern.findall(text)

        # Find percentage/dollar/metric patterns
        metric_pattern = re.compile(r"\$[\d,.]+[MBK]?\b|\b\d+(?:\.\d+)?%\b")
        metrics = metric_pattern.findall(text)

        entities = []
        seen = set()
        for name in names + roles:
            name = name.strip()
            if len(name) > 3 and name not in seen:
                seen.add(name)
                entities.append(
                    {
                        "name": name,
                        "type": "EXTRACTED",
                        "description": f"Mentioned in {doc.path.name}",
                    }
                )
        for m in metrics[:5]:
            entities.append(
                {
                    "name": m.strip(),
                    "type": "METRIC",
                    "description": f"Metric from {doc.path.name}",
                }
            )

        # Generate basic relationships between co-occurring entities
        relationships = []
        entity_names = [e["name"] for e in entities]
        for i, e1 in enumerate(entity_names[:10]):
            for e2 in entity_names[i + 1 : i + 4]:
                relationships.append(
                    {
                        "source": e1,
                        "target": e2,
                        "type": "co_occurs_with",
                        "context": f"Both mentioned in {doc.path.name}",
                    }
                )

        return entities, relationships

    # ── Community Summarization ────────────────────────────────────────────

    async def _generate_community_summaries(self) -> None:
        """Generate real LLM summaries for each community."""
        if not self.communities:
            return

        for comm_idx, members in self.communities.items():
            if comm_idx in self.community_summaries:
                continue  # Already summarized

            # Build a text description of the community
            member_list = list(members)[:20]
            parts = []
            for member in member_list:
                desc = self.entity_descriptions.get(member, "")
                line = f"- {member}"
                if desc:
                    line += f": {desc}"
                parts.append(line)

            # Add key relationships within the community
            rel_parts = []
            for m1 in member_list[:10]:
                if not self.graph.has_node(m1):
                    continue
                for m2 in self.graph.neighbors(m1):
                    if m2 in members:
                        edge = self.graph[m1][m2]
                        rel_type = edge.get("type", "related_to")
                        rel_parts.append(f"- {m1} → {rel_type} → {m2}")

            community_desc = "\n".join(parts)
            if rel_parts:
                community_desc += "\n\nRelationships:\n" + "\n".join(rel_parts[:15])

            llm = self.llm
            if llm is not None and len(member_list) >= 2:
                try:
                    summary = await llm.agenerate(
                        f"Summarize this group of related entities and their "
                        f"relationships in 2-3 sentences. Focus on WHO they are, "
                        f"WHAT they do, and HOW they relate to each other.\n\n"
                        f"Entities:\n{community_desc}",
                        system="You are a concise knowledge graph summarizer.",
                    )
                    self.community_summaries[comm_idx] = summary
                except Exception as e:
                    logger.warning(f"Failed to summarize community {comm_idx}: {e}")
                    self.community_summaries[
                        comm_idx
                    ] = f"Group containing: {', '.join(member_list[:10])}"
            else:
                self.community_summaries[
                    comm_idx
                ] = f"Group containing: {', '.join(member_list[:10])}"
