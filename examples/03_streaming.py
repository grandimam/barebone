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
    )

    async for event in agent.stream("Write a haiku about Python."):
        if event["type"] == "text_delta":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print()


if __name__ == "__main__":
    asyncio.run(main())
