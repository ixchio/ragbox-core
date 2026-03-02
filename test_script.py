import sys
from ragbox import RAGBox

print("Initializing RAGBox...")
rag = RAGBox("./test_docs")

print("Querying RAGBox...")
answer = rag.query("What's our vacation policy?")

print("Answer received:")
print(answer)
