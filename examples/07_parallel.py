import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def main():
    topics = ["Python", "Rust", "Go"]

    async def describe(topic: str) -> tuple[str, str]:
        agent = Agent(api_key=API_KEY, model="claude-sonnet-4-20250514")
        response = await agent.run(f"Describe {topic} in one sentence.")
        return topic, response.content

    results = await asyncio.gather(*[describe(t) for t in topics])

    for topic, description in results:
        print(f"{topic}: {description}")


if __name__ == "__main__":
    asyncio.run(main())
