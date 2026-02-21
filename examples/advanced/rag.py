"""RAG (Retrieval-Augmented Generation) examples."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"

# Sample document store
DOCUMENTS = [
    {
        "id": 1,
        "title": "Python Basics",
        "content": "Python is a high-level programming language. It uses indentation for code blocks. Variables don't need type declarations.",
    },
    {
        "id": 2,
        "title": "Python Functions",
        "content": "Functions in Python are defined with 'def'. They can have default arguments and return multiple values using tuples.",
    },
    {
        "id": 3,
        "title": "Python Classes",
        "content": "Classes in Python use 'class' keyword. The __init__ method is the constructor. Self refers to the instance.",
    },
    {
        "id": 4,
        "title": "Python Async",
        "content": "Async programming uses async/await syntax. asyncio is the standard library for async IO. Coroutines are defined with 'async def'.",
    },
    {
        "id": 5,
        "title": "Error Handling",
        "content": "Python uses try/except for error handling. Finally blocks always execute. You can raise custom exceptions.",
    },
]


def simple_search(query: str, top_k: int = 2) -> list[dict]:
    """Simple keyword-based document search."""
    query_words = set(query.lower().split())
    scored = []

    for doc in DOCUMENTS:
        content_words = set(doc["content"].lower().split())
        title_words = set(doc["title"].lower().split())
        score = len(query_words & content_words) + len(query_words & title_words) * 2
        if score > 0:
            scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]


async def basic_rag(question: str) -> str:
    """Basic RAG: retrieve relevant docs, then answer with context."""
    print("=" * 60)
    print("Basic RAG")
    print("=" * 60)

    docs = simple_search(question)
    print(f"Retrieved {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc['title']}")

    context = "\n\n".join(f"## {d['title']}\n{d['content']}" for d in docs)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)
    response = await agent.run(f"""Answer the question based on the provided context.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}""")

    print(f"\nAnswer: {response.content}")
    return response.content


async def rag_with_reranking(question: str) -> str:
    """RAG with LLM-based reranking of retrieved documents."""
    print("\n" + "=" * 60)
    print("RAG with Reranking")
    print("=" * 60)

    docs = simple_search(question, top_k=4)
    print(f"Retrieved {len(docs)} candidates")

    docs_text = "\n".join(
        f"{i + 1}. {d['title']}: {d['content'][:100]}..." for i, d in enumerate(docs)
    )

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    # Rerank with LLM
    response = await agent.run(f"""Rank these documents by relevance to the question.
Return the numbers of the top 2 most relevant, comma-separated.

Question: {question}

Documents:
{docs_text}""")

    try:
        indices = [int(x.strip()) - 1 for x in response.content.split(",")]
        reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
    except (ValueError, IndexError):
        reranked = docs[:2]

    print(f"Reranked top docs: {[d['title'] for d in reranked]}")

    context = "\n\n".join(f"## {d['title']}\n{d['content']}" for d in reranked)

    response = await agent.run(f"""Answer based on context:

{context}

Question: {question}""")

    print(f"\nAnswer: {response.content}")
    return response.content


async def rag_with_query_expansion(question: str) -> str:
    """RAG with query expansion for better retrieval."""
    print("\n" + "=" * 60)
    print("RAG with Query Expansion")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    # Expand query
    response = await agent.run(f"""Generate 2 alternative phrasings of this question for search:

{question}

Return just the alternatives, one per line.""")
    expansions = [question] + response.content.strip().split("\n")
    print(f"Expanded queries: {expansions}")

    # Retrieve with all queries
    all_docs = []
    seen_ids = set()
    for q in expansions:
        for doc in simple_search(q, top_k=2):
            if doc["id"] not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc["id"])

    print(f"Retrieved {len(all_docs)} unique documents")

    context = "\n\n".join(f"## {d['title']}\n{d['content']}" for d in all_docs)

    response = await agent.run(f"""Answer based on context:

{context}

Question: {question}""")

    print(f"\nAnswer: {response.content}")
    return response.content


async def main():
    await basic_rag("How do I define a function in Python?")
    await rag_with_reranking("What is async programming?")
    await rag_with_query_expansion("How do I handle errors?")


if __name__ == "__main__":
    asyncio.run(main())
