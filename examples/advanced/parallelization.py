"""Parallelization examples - running multiple LLM calls concurrently."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def parallel_calls():
    """Run multiple independent calls in parallel."""
    print("=" * 60)
    print("Parallel Calls")
    print("=" * 60)

    topics = ["Python", "Rust", "Go"]

    async def describe_topic(topic: str) -> tuple[str, str]:
        provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
        agent = Agent(provider=provider)
        response = await agent.run(f"Describe {topic} in one sentence.")
        return topic, response.content

    tasks = [describe_topic(topic) for topic in topics]
    results = await asyncio.gather(*tasks)

    for topic, description in results:
        print(f"{topic}: {description}")


async def map_reduce():
    """Map-reduce pattern: process documents in parallel, then combine."""
    print("\n" + "=" * 60)
    print("Map-Reduce")
    print("=" * 60)

    documents = [
        "AI is transforming healthcare with diagnostic tools.",
        "Machine learning models can predict patient outcomes.",
        "Robots are assisting in surgical procedures.",
    ]

    async def summarize_doc(doc: str) -> str:
        provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
        agent = Agent(provider=provider)
        response = await agent.run(f"Extract the key point in 5 words or less: {doc}")
        return response.content

    # Map: summarize each document in parallel
    tasks = [summarize_doc(doc) for doc in documents]
    summaries = await asyncio.gather(*tasks)

    print("Summaries:")
    for s in summaries:
        print(f"  - {s}")

    # Reduce: combine summaries
    combined = "\n".join(summaries)
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)
    response = await agent.run(f"Combine these points into one sentence:\n{combined}")
    print(f"\nCombined: {response.content}")


async def voting():
    """Voting/Best-of-N pattern: generate multiple candidates, pick the best."""
    print("\n" + "=" * 60)
    print("Voting / Best-of-N")
    print("=" * 60)

    question = "What's a creative name for a coffee shop?"

    async def generate_candidate() -> str:
        provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
        agent = Agent(provider=provider, temperature=1.0)
        response = await agent.run(f"{question} Give just the name, nothing else.")
        return response.content

    # Generate multiple candidates in parallel
    tasks = [generate_candidate() for _ in range(3)]
    candidates = await asyncio.gather(*tasks)

    print("Candidates:")
    for c in candidates:
        print(f"  - {c}")

    # Pick the best
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)
    response = await agent.run(
        f"Pick the best coffee shop name and explain why in one sentence:\n{chr(10).join(candidates)}"
    )
    print(f"\nWinner: {response.content}")


async def main():
    await parallel_calls()
    await map_reduce()
    await voting()


if __name__ == "__main__":
    asyncio.run(main())
