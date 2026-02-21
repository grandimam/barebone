"""Tool loop example demonstrating agentic tool execution."""

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
def calculator(expression: str) -> str:
    """Perform arithmetic calculations."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_fact(number: int) -> str:
    """Get a fact about a number."""
    facts = {
        42: "The answer to life, the universe, and everything",
        7: "Considered lucky in many cultures",
        13: "Considered unlucky in Western culture",
    }
    return facts.get(number, f"{number} is just a number")


async def tool_loop_example():
    """Demonstrate the agent's tool loop."""
    print("=" * 60)
    print("Tool Loop Example")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        tools=[calculator, get_fact],
        system="You are a helpful math assistant. Use tools to help with calculations.",
    )

    # The agent will automatically loop: call tool -> get result -> continue
    response = await agent.run("What is (15 + 27) * 3?")
    print(f"\nResult: {response.content}")

    # Show the conversation history
    print(f"\nMessage history ({len(agent.messages)} messages):")
    for i, msg in enumerate(agent.messages):
        if msg.tool_calls:
            print(f"  {i+1}. {msg.role}: [tool calls: {[tc.name for tc in msg.tool_calls]}]")
        elif msg.tool_results:
            print(f"  {i+1}. {msg.role}: [tool results]")
        else:
            content = msg.content[:50] + "..." if msg.content and len(msg.content) > 50 else msg.content
            print(f"  {i+1}. {msg.role}: {content}")


async def multi_tool_example():
    """Multiple tools working together."""
    print("\n" + "=" * 60)
    print("Multi-Tool Example")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        tools=[calculator, get_fact],
        system="You are a helpful assistant with math and trivia knowledge.",
    )

    response = await agent.run(
        "What is 6 * 7, and can you tell me an interesting fact about that number?"
    )
    print(f"\nResult: {response.content}")


async def main():
    await tool_loop_example()
    await multi_tool_example()


if __name__ == "__main__":
    asyncio.run(main())
