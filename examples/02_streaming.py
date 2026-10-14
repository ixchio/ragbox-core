"""
Example 2: Streaming Answers Token by Token

Every LLM provider (OpenAI, Anthropic, Groq) streams automatically.
"""
import asyncio
from ragbox import RAGBox


async def main():
    rag = RAGBox("./docs")

    print("Streaming answer: ", end="")
    async for chunk in rag.astream("Summarize all the key findings"):
        print(chunk, end="", flush=True)
    print()  # newline at end


asyncio.run(main())
