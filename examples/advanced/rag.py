"""
RAG (Retrieval-Augmented Generation) pattern.

Retrieve relevant context, augment prompt, generate response.
"""

from barebone import complete, user


# Simulated document store
DOCUMENTS = [
    {"id": 1, "title": "Python Basics", "content": "Python is a high-level programming language. It uses indentation for code blocks. Variables don't need type declarations."},
    {"id": 2, "title": "Python Functions", "content": "Functions in Python are defined with 'def'. They can have default arguments and return multiple values using tuples."},
    {"id": 3, "title": "Python Classes", "content": "Classes in Python use 'class' keyword. The __init__ method is the constructor. Self refers to the instance."},
    {"id": 4, "title": "Python Async", "content": "Async programming uses async/await syntax. asyncio is the standard library for async IO. Coroutines are defined with 'async def'."},
    {"id": 5, "title": "Error Handling", "content": "Python uses try/except for error handling. Finally blocks always execute. You can raise custom exceptions."},
]


def simple_search(query: str, top_k: int = 2) -> list[dict]:
    """Simple keyword search (replace with vector search in production)."""
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


def basic_rag(question: str) -> str:
    """Basic RAG: retrieve, augment, generate."""
    print("=" * 60)
    print("Basic RAG")
    print("=" * 60)

    # Retrieve
    docs = simple_search(question)
    print(f"Retrieved {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc['title']}")

    # Augment
    context = "\n\n".join(f"## {d['title']}\n{d['content']}" for d in docs)

    # Generate
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Answer the question based on the provided context.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}""")
    ])

    print(f"\nAnswer: {response.content}")
    return response.content


def rag_with_reranking(question: str) -> str:
    """RAG with LLM-based reranking."""
    print("\n" + "=" * 60)
    print("RAG with Reranking")
    print("=" * 60)

    # Retrieve more candidates
    docs = simple_search(question, top_k=4)
    print(f"Retrieved {len(docs)} candidates")

    # Rerank using LLM
    docs_text = "\n".join(f"{i+1}. {d['title']}: {d['content'][:100]}..." for i, d in enumerate(docs))

    response = complete("claude-sonnet-4-20250514", [
        user(f"""Rank these documents by relevance to the question.
Return the numbers of the top 2 most relevant, comma-separated.

Question: {question}

Documents:
{docs_text}""")
    ])

    try:
        indices = [int(x.strip()) - 1 for x in response.content.split(",")]
        reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
    except (ValueError, IndexError):
        reranked = docs[:2]

    print(f"Reranked top docs: {[d['title'] for d in reranked]}")

    # Generate with reranked context
    context = "\n\n".join(f"## {d['title']}\n{d['content']}" for d in reranked)

    response = complete("claude-sonnet-4-20250514", [
        user(f"""Answer based on context:

{context}

Question: {question}""")
    ])

    print(f"\nAnswer: {response.content}")
    return response.content


def rag_with_query_expansion(question: str) -> str:
    """Expand query to improve retrieval."""
    print("\n" + "=" * 60)
    print("RAG with Query Expansion")
    print("=" * 60)

    # Expand query
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Generate 2 alternative phrasings of this question for search:

{question}

Return just the alternatives, one per line.""")
    ])
    expansions = [question] + response.content.strip().split("\n")
    print(f"Expanded queries: {expansions}")

    # Retrieve for all queries
    all_docs = []
    seen_ids = set()
    for q in expansions:
        for doc in simple_search(q, top_k=2):
            if doc["id"] not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc["id"])

    print(f"Retrieved {len(all_docs)} unique documents")

    # Generate
    context = "\n\n".join(f"## {d['title']}\n{d['content']}" for d in all_docs)

    response = complete("claude-sonnet-4-20250514", [
        user(f"""Answer based on context:

{context}

Question: {question}""")
    ])

    print(f"\nAnswer: {response.content}")
    return response.content


if __name__ == "__main__":
    basic_rag("How do I define a function in Python?")
    rag_with_reranking("What is async programming?")
    rag_with_query_expansion("How do I handle errors?")
