"""
Example 3: Cross-Document GraphRAG

RAGBox automatically extracts entities and relationships across all your documents
and routes complex queries through the Knowledge Graph.

No config. No graph DB setup. Just works.
"""
from ragbox import RAGBox

rag = RAGBox("./company-docs")

# Simple factual — goes to vector search
print(rag.query("What is our code review policy?"))

# Cross-document relationship — automatically goes to GraphRAG
print(rag.query("Who does Maria Santos report to?"))

# Multi-hop — needs 3+ documents connected
print(rag.query("How did the platform team outage affect Q4 revenue?"))

# Chronological reasoning across docs
print(rag.query("What changed in the security policy after the November incident?"))
