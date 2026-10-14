"""
Example 1: Synchronous document query

Minimal synchronous query example using the public Python API.
"""
from ragbox import RAGBox

rag = RAGBox("./docs")
print(rag.query("What is our vacation policy?"))
