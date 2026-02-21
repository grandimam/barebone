"""Example demonstrating custom tools."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider
from barebone import tool

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def get_fact(number: int) -> str:
    """Get an interesting fact about a number."""
    facts = {
        42: "The answer to life, the universe, and everything",
        7: "Considered lucky in many cultures",
        13: "Considered unlucky in Western culture",
        100: "A perfect score, often called a century",
    }
    return facts.get(number, f"{number} is just a number")


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    provider = AnthropicProvider(api_key=api_key)

    agent = Agent(
        provider=provider,
        tools=[add, multiply, get_fact],
        system="You are a helpful math assistant. Use the tools to help with calculations.",
    )

    # Simple addition
    print("=" * 50)
    print("Addition Example")
    print("=" * 50)
    response = await agent.run("What is 15 + 27?")
    print(response.content)

    # Chained operations
    print("\n" + "=" * 50)
    print("Chained Operations")
    print("=" * 50)
    response = await agent.run("What is 6 * 7, and tell me a fact about the result?")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
