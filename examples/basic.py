"""Basic agent example with tools."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider
from barebone import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    provider = AnthropicProvider(api_key=api_key)

    agent = Agent(
        provider=provider,
        tools=[get_weather, calculate],
        system="You are a helpful assistant. Be concise.",
    )

    # Weather query with tool use
    response = await agent.run("What's the weather in Tokyo?")
    print(f"Weather: {response.content}")

    # Math query with tool use
    response = await agent.run("What is 123 * 456?")
    print(f"Math: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
