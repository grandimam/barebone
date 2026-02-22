import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def main():
    agent = Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Remember our conversation.",
    )

    response = await agent.run("My name is Alice.")
    print(f"Turn 1: {response.content}")

    response = await agent.run("What's my name?")
    print(f"Turn 2: {response.content}")

    print(f"\nConversation history: {len(agent.messages)} messages")


if __name__ == "__main__":
    asyncio.run(main())
