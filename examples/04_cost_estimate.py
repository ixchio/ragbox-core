"""
Example 4: Know the cost BEFORE it runs.

RAGBox estimates the $ cost of indexing your documents before it touches any API.
"""
from ragbox import RAGBox

rag = RAGBox("./large-docs")

# Get the estimate — no API calls yet
estimate = rag.estimate_cost()

print(f"Documents:    {estimate.num_documents}")
print(f"Total tokens: {estimate.input_tokens:,}")
print(f"Estimated cost: ${estimate.total_cost_usd:.4f}")

# Proceed only if under budget
if estimate.total_cost_usd < 0.50:
    print("\nUnder budget — indexing now...")
    # rag.query(...)  # triggers actual indexing on first query
else:
    print(
        f"\nOver budget (${estimate.total_cost_usd:.2f}). Reduce docs or switch to Groq."
    )
