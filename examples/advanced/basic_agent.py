"""Basic agent examples demonstrating core functionality."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider
from barebone import tool

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


@tool
def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}! Nice to meet you."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


async def basic_example():
    """Basic agent with tools."""
    print("=" * 60)
    print("Basic Agent Example")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="You are a helpful assistant. Be concise.",
        tools=[greet_user, calculate],
    )

    response = await agent.run("Greet Alice and then calculate 15 * 7.")
    print(f"\nResponse: {response.content}")


async def streaming_example():
    """Streaming responses."""
    print("\n" + "=" * 60)
    print("Streaming Example")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="You are a helpful assistant.",
    )

    print("\nStreaming response:")
    async for event in provider.stream(
        messages=agent.messages + [{"role": "user", "content": "Write a haiku about coding."}],
        system=agent.system,
    ):
        if event.get("type") == "text_delta":
            print(event.get("text", ""), end="", flush=True)
        elif event.get("type") == "done":
            print()


async def conversation_example():
    """Multi-turn conversation with memory."""
    print("\n" + "=" * 60)
    print("Conversation Example")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="You are a helpful assistant. Remember our conversation.",
    )

    response = await agent.run("My name is Alice.")
    print(f"Turn 1: {response.content}")

    response = await agent.run("What's my name?")
    print(f"Turn 2: {response.content}")

    # Clear messages and ask again
    agent.messages.clear()
    response = await agent.run("What's my name?")
    print(f"Turn 3 (after clear): {response.content}")


async def main():
    await basic_example()
    await streaming_example()
    await conversation_example()


if __name__ == "__main__":
    asyncio.run(main())
