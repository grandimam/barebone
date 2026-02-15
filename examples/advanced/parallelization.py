"""
Parallelization pattern.

Run multiple LLM calls concurrently and aggregate results.
"""

import asyncio

from barebone import acomplete, user


async def parallel_calls():
    """Run multiple independent calls in parallel."""
    print("=" * 60)
    print("Parallel Calls")
    print("=" * 60)

    topics = ["Python", "Rust", "Go"]

    # Run all calls concurrently
    tasks = [
        acomplete("claude-sonnet-4-20250514", [
            user(f"Describe {topic} in one sentence.")
        ])
        for topic in topics
    ]
    responses = await asyncio.gather(*tasks)

    for topic, response in zip(topics, responses):
        print(f"{topic}: {response.content}")


async def map_reduce():
    """Map operation across inputs, then reduce to single output."""
    print("\n" + "=" * 60)
    print("Map-Reduce")
    print("=" * 60)

    documents = [
        "AI is transforming healthcare with diagnostic tools.",
        "Machine learning models can predict patient outcomes.",
        "Robots are assisting in surgical procedures.",
    ]

    # Map: Summarize each document
    tasks = [
        acomplete("claude-sonnet-4-20250514", [
            user(f"Extract the key point in 5 words or less: {doc}")
        ])
        for doc in documents
    ]
    summaries = await asyncio.gather(*tasks)
    summaries = [r.content for r in summaries]

    print("Summaries:")
    for s in summaries:
        print(f"  - {s}")

    # Reduce: Combine summaries
    combined = "\n".join(summaries)
    response = await acomplete("claude-sonnet-4-20250514", [
        user(f"Combine these points into one sentence:\n{combined}")
    ])
    print(f"\nCombined: {response.content}")


async def voting():
    """Get multiple responses and pick the best."""
    print("\n" + "=" * 60)
    print("Voting / Best-of-N")
    print("=" * 60)

    question = "What's a creative name for a coffee shop?"

    # Generate multiple candidates
    tasks = [
        acomplete("claude-sonnet-4-20250514", [
            user(f"{question} Give just the name, nothing else.")
        ], temperature=1.0)
        for _ in range(3)
    ]
    responses = await asyncio.gather(*tasks)
    candidates = [r.content for r in responses]

    print("Candidates:")
    for c in candidates:
        print(f"  - {c}")

    # Vote on best
    response = await acomplete("claude-sonnet-4-20250514", [
        user(f"Pick the best coffee shop name and explain why in one sentence:\n{chr(10).join(candidates)}")
    ])
    print(f"\nWinner: {response.content}")


if __name__ == "__main__":
    asyncio.run(parallel_calls())
    asyncio.run(map_reduce())
    asyncio.run(voting())
