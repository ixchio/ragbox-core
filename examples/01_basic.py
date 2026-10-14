"""
Example 1: The 3-Line RAG App

Compare this with the LangChain equivalent in ../COMPARISON.md — 31 lines vs 3.
"""
from ragbox import RAGBox

rag = RAGBox("./docs")
print(rag.query("What is our vacation policy?"))
