import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import tool

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@tool
def search(query: str) -> str:
    return f"Search results for '{query}': Found 3 relevant articles."


@tool
def read_article(title: str) -> str:
    return f"Article '{title}' content: This is the article content."


@tool
def write_summary(content: str) -> str:
    return f"Summary written: {content[:50]}..."


async def main():
    agent = Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
        tools=[search, read_article, write_summary],
        system="""You are a research assistant. When given a task:
1. First create a plan
2. Execute each step
3. Provide final results""",
    )

    response = await agent.run("Research the latest trends in AI and summarize them.")
    print(f"Result: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
