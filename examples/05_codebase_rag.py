"""
Example 5: Chat with a codebase.

RAGBox auto-parses Python files using AST, so it understands
class structures, function relationships, and imports — not just text.

Great for:
- "How does the auth middleware work?"
- "Which functions handle database writes?"
- "What does UserService depend on?"
"""
from ragbox import RAGBox

# Point at your Python project
rag = RAGBox("./src")

questions = [
    "How does authentication work?",
    "Which functions call the database?",
    "What is the UserService responsible for?",
    "Explain the error handling pattern used throughout this codebase.",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag.query(q)}")
