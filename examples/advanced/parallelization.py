import asyncio
import os

from barebone import acomplete
from barebone import user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


async def parallel_calls():
    print("=" * 60)
    print("Parallel Calls")
    print("=" * 60)

    topics = ["Python", "Rust", "Go"]

    tasks = [
        acomplete(MODEL, [user(f"Describe {topic} in one sentence.")], api_key=API_KEY)
        for topic in topics
    ]
    responses = await asyncio.gather(*tasks)

    for topic, response in zip(topics, responses):
        print(f"{topic}: {response.content}")


async def map_reduce():
    print("\n" + "=" * 60)
    print("Map-Reduce")
    print("=" * 60)

    documents = [
        "AI is transforming healthcare with diagnostic tools.",
        "Machine learning models can predict patient outcomes.",
        "Robots are assisting in surgical procedures.",
    ]

    tasks = [
        acomplete(
            MODEL, [user(f"Extract the key point in 5 words or less: {doc}")], api_key=API_KEY
        )
        for doc in documents
    ]
    summaries = await asyncio.gather(*tasks)
    summaries = [r.content for r in summaries]

    print("Summaries:")
    for s in summaries:
        print(f"  - {s}")

    combined = "\n".join(summaries)
    response = await acomplete(
        MODEL, [user(f"Combine these points into one sentence:\n{combined}")], api_key=API_KEY
    )
    print(f"\nCombined: {response.content}")


async def voting():
    print("\n" + "=" * 60)
    print("Voting / Best-of-N")
    print("=" * 60)

    question = "What's a creative name for a coffee shop?"

    tasks = [
        acomplete(
            MODEL,
            [user(f"{question} Give just the name, nothing else.")],
            api_key=API_KEY,
            temperature=1.0,
        )
        for _ in range(3)
    ]
    responses = await asyncio.gather(*tasks)
    candidates = [r.content for r in responses]

    print("Candidates:")
    for c in candidates:
        print(f"  - {c}")

    response = await acomplete(
        MODEL,
        [
            user(
                f"Pick the best coffee shop name and explain why in one sentence:\n{chr(10).join(candidates)}"
            )
        ],
        api_key=API_KEY,
    )
    print(f"\nWinner: {response.content}")


if __name__ == "__main__":
    asyncio.run(parallel_calls())
    asyncio.run(map_reduce())
    asyncio.run(voting())
