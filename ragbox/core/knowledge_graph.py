"""
Layer 4: KNOWLEDGE GRAPH
Entity extraction, Leiden/Louvain community detection, real graph traversal,
and community summarization for cross-document reasoning.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional
import asyncio
from loguru import logger
import networkx as nx
import uuid
import igraph as ig
import leidenalg
import community as community_louvain  # python-louvain

from ragbox.models.documents import Document
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

    def add_document(
        self,
        doc_id: str,
        entities: List[str],
        relationships: List[Dict[str, Any]],
        entity_descriptions: Optional[Dict[str, str]] = None,
        entity_source_texts: Optional[Dict[str, str]] = None,
    ) -> None:
        new_nodes = 0
        new_edges = 0

        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, source_docs=set(), weight=1.0)
                new_nodes += 1
            else:
                self.graph.nodes[entity]["weight"] = (
                    self.graph.nodes[entity].get("weight", 1.0) + 0.1
                )
            self.graph.nodes[entity]["source_docs"].add(doc_id)

            # Store entity descriptions and source text
            if entity_descriptions and entity in entity_descriptions:
                self.entity_descriptions[entity] = entity_descriptions[entity]
            if entity_source_texts and entity in entity_source_texts:
                self.entity_source_texts.setdefault(entity, []).append(
                    entity_source_texts[entity]
                )

        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "related_to")
            context = rel.get("context", "")

            if source and target:
                if self.graph.has_edge(source, target):
                    self.graph[source][target]["weight"] += 1.0
                    self.graph[source][target]["docs"].add(doc_id)
                    # Accumulate relationship types
                    existing_types = self.graph[source][target].get("types", set())
                    existing_types.add(rel_type)
                    self.graph[source][target]["types"] = existing_types
                else:
                    self.graph.add_edge(
                        source, target,
                        type=rel_type, types={rel_type},
                        weight=1.0, docs={doc_id},
                    )
                    new_edges += 1

                # Store relationship context
                if context:
                    key = (source, target)
                    self.relation_contexts.setdefault(key, []).append(context)

        self.pending_updates += new_nodes + new_edges

        ratio = self.pending_updates / max(1, len(self.graph.nodes))
        if self.pending_updates > self.optimize_threshold or ratio > 0.1:
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
            ig_graph, leidenalg.ModularityVertexPartition,
            weights=weights, n_iterations=2,
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

        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        scored: Dict[str, float] = {}

        for entity in self.graph.nodes():
            entity_lower = entity.lower()
            entity_tokens = set(entity_lower.split())
            score = 0.0

            # Exact substring match (entity appears literally in query)
            if entity_lower in query_lower:
                score += 1.0
            # Entity name contains a query word
            elif query_tokens & entity_tokens:
                overlap = len(query_tokens & entity_tokens)
                score += 0.5 * overlap / max(len(entity_tokens), 1)

            # Check description overlap
            desc = self.entity_descriptions.get(entity, "").lower()
            if desc:
                desc_tokens = set(desc.split())
                desc_overlap = len(query_tokens & desc_tokens)
                if desc_overlap > 0:
                    score += 0.3 * desc_overlap / max(len(query_tokens), 1)

            if score > 0:
                # Boost by graph centrality (well-connected entities are more important)
                degree = self.graph.degree(entity)
                score *= 1.0 + 0.05 * min(degree, 10)
                scored[entity] = score

        results = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return results[:20]

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
                    for key in [(entity, neighbor), (neighbor, entity)]:
                        contexts = self.relation_contexts.get(key, [])
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
        """
        Search the knowledge graph with real entity matching and traversal.

        1. Match query terms to graph entities (fuzzy + token overlap)
        2. Expand matched entities via graph traversal (neighbors + community)
        3. Collect entity descriptions, source texts, and relationship context
        4. Return structured result with synthesized context
        """
        if not self.graph.nodes:
            return GraphQueryResult(
                relevant_entities=[], relevant_relations=[],
                relevant_communities=[], synthesized_context="",
            )

        # Step 1: Match query to entities
        matched = self._match_query_to_entities(query)
        if not matched:
            return GraphQueryResult(
                relevant_entities=[], relevant_relations=[],
                relevant_communities=[], synthesized_context="",
            )

        seed_entities = [name for name, _ in matched[:10]]
        logger.info(
            f"Graph query matched {len(seed_entities)} seed entities: "
            f"{seed_entities[:5]}"
        )

        # Step 2: Expand via graph traversal
        expanded = set(seed_entities)
        for entity in seed_entities[:5]:
            related = self.get_related_entities(entity, max_distance=2)
            expanded.update(related[:8])

        expanded_list = list(expanded)

        # Step 3: Collect rich context
        synthesized = self._collect_entity_context(expanded_list)

        # Step 4: Build structured result
        result_entities = []
        for name in expanded_list:
            desc = self.entity_descriptions.get(name, "")
            result_entities.append(Entity(
                id=name, name=name, entity_type="EXTRACTED", description=desc,
            ))

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
                    result_relations.append(Relation(
                        source_id=entity, target_id=neighbor,
                        relation_type=edge.get("type", "related_to"),
                        context="; ".join(
                            self.relation_contexts.get((entity, neighbor), [])[:2]
                        ),
                    ))

        touched_communities = set()
        for entity in expanded_list:
            comm = self.get_community_for_entity(entity)
            if comm is not None:
                touched_communities.add(comm)

        result_communities = []
        for comm_idx in list(touched_communities)[:5]:
            summary = self.community_summaries.get(comm_idx, "")
            members = list(self.communities.get(comm_idx, set()))
            result_communities.append(Community(
                id=f"community_{comm_idx}", level=1,
                entity_ids=members, summary=summary,
            ))

        return GraphQueryResult(
            relevant_entities=result_entities,
            relevant_relations=result_relations,
            relevant_communities=result_communities,
            synthesized_context=synthesized,
        )

    # ── Build Pipeline ────────────────────────────────────────────────────

    async def build_from_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logger.info(
            f"Extracting entities and relations for {len(documents)} documents."
        )

        for doc in documents:
            entities_data, relations_data = await self._extract_single_doc(doc)
            if not entities_data:
                continue

            entity_names = [e["name"] for e in entities_data]
            entity_descs = {e["name"]: e.get("description", "") for e in entities_data}

            # Store a text snippet as source context for each entity
            doc_snippet = doc.content[:2000]
            entity_sources = {name: doc_snippet for name in entity_names}

            rels = [
                {
                    "source": r["source"], "target": r["target"],
                    "type": r.get("type", "RELATED_TO"),
                    "context": r.get("context", ""),
                }
                for r in relations_data
            ]

            self.add_document(
                doc_id=doc.id, entities=entity_names,
                relationships=rels,
                entity_descriptions=entity_descs,
                entity_source_texts=entity_sources,
            )

        # Community detection + summarization
        logger.info("Running community detection and summarization.")
        self._optimize_graph()
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

        if not getattr(self, "llm", None):
            logger.warning("No LLM client — using regex-based entity extraction.")
            return self._extract_entities_regex(doc)

        cache_dir = Path(".ragbox_state/graph_cache")
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
                {"name": "string", "type": "PERSON|ORG|CONCEPT|METRIC|EVENT|POLICY|TEAM|ROLE|PRODUCT", "description": "string"}
            ],
            "relationships": [
                {"source": "entity name", "target": "entity name", "type": "string", "context": "brief description of relationship"}
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
                res = await self.llm.agenerate_structured(
                    prompt, schema,
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

    def _extract_entities_regex(
        self, doc: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Regex-based entity extraction fallback. Finds capitalized proper nouns
        and common named entity patterns. No LLM needed.
        """
        import re

        text = doc.content[:5000]

        # Find capitalized multi-word names (e.g. "Maria Santos", "VP of Engineering")
        name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:of|the|and|for|in)\s+)?(?:[A-Z][a-z]+)+)\b")
        names = name_pattern.findall(text)

        # Find role/title patterns
        role_pattern = re.compile(r"\b((?:VP|CEO|CTO|CFO|COO|Director|Manager|Lead|Head|Chief)\s+(?:of\s+)?[\w\s]{2,30})\b")
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
                entities.append({
                    "name": name, "type": "EXTRACTED",
                    "description": f"Mentioned in {doc.path.name}",
                })
        for m in metrics[:5]:
            entities.append({
                "name": m.strip(), "type": "METRIC",
                "description": f"Metric from {doc.path.name}",
            })

        # Generate basic relationships between co-occurring entities
        relationships = []
        entity_names = [e["name"] for e in entities]
        for i, e1 in enumerate(entity_names[:10]):
            for e2 in entity_names[i + 1 : i + 4]:
                relationships.append({
                    "source": e1, "target": e2,
                    "type": "co_occurs_with",
                    "context": f"Both mentioned in {doc.path.name}",
                })

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

            if getattr(self, "llm", None) and len(member_list) >= 2:
                try:
                    summary = await self.llm.agenerate(
                        f"Summarize this group of related entities and their "
                        f"relationships in 2-3 sentences. Focus on WHO they are, "
                        f"WHAT they do, and HOW they relate to each other.\n\n"
                        f"Entities:\n{community_desc}",
                        system="You are a concise knowledge graph summarizer.",
                    )
                    self.community_summaries[comm_idx] = summary
                except Exception as e:
                    logger.warning(f"Failed to summarize community {comm_idx}: {e}")
                    self.community_summaries[comm_idx] = (
                        f"Group containing: {', '.join(member_list[:10])}"
                    )
            else:
                self.community_summaries[comm_idx] = (
                    f"Group containing: {', '.join(member_list[:10])}"
                )
