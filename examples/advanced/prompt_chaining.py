"""Prompt chaining examples - sequential LLM calls building on each other."""

import asyncio
import os
from collections.abc import Callable

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def chain(*prompts: str | Callable[[str], str]) -> str:
    """Chain multiple prompts, passing each result to the next."""
    result = ""
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    for prompt in prompts:
        if callable(prompt):
            prompt = prompt(result)
        agent = Agent(provider=provider)
        response = await agent.run(prompt)
        result = response.content

    return result


async def basic_chaining():
    """Basic example of sequential prompt chaining."""
    print("=" * 60)
    print("Basic Prompt Chaining")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    # Step 1: Generate outline
    agent = Agent(provider=provider)
    response = await agent.run("Create a 3-point outline for an article about AI safety")
    outline = response.content
    print(f"Outline:\n{outline}\n")

    # Step 2: Expand outline
    response = await agent.run(f"Expand this outline into a short article:\n\n{outline}")
    article = response.content
    print(f"Article:\n{article}\n")

    # Step 3: Generate title
    response = await agent.run(f"Write a compelling title for this article:\n\n{article}")
    print(f"Title: {response.content}")


async def chain_helper_example():
    """Using the chain helper function."""
    print("\n" + "=" * 60)
    print("Chain Helper Function")
    print("=" * 60)

    result = await chain(
        "List 3 startup ideas in fintech. Just the names.",
        lambda ideas: f"Pick the best idea and explain why:\n{ideas}",
        lambda pick: f"Write a one-paragraph pitch:\n{pick}",
    )
    print(result)


async def gate_and_branch():
    """Conditional branching based on intermediate results."""
    print("\n" + "=" * 60)
    print("Gate and Branch")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    query = "Is Python good for machine learning?"

    # Gate: check if query is technical
    response = await agent.run(f"Is this a technical question? Reply YES or NO only: {query}")
    is_technical = "yes" in response.content.lower()

    print(f"Query: {query}")
    print(f"Technical: {is_technical}")

    # Branch based on result
    if is_technical:
        response = await agent.run(
            f"As a technical expert, answer: {query}",
        )
    else:
        response = await agent.run(
            f"As a general assistant, answer: {query}",
        )

    print(f"Response: {response.content}")


async def main():
    await basic_chaining()
    await chain_helper_example()
    await gate_and_branch()


if __name__ == "__main__":
    asyncio.run(main())
