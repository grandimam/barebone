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

    response = await agent.run("Create a 3-point outline for an article about AI safety")
    outline = response.content
    print(f"Outline:\n{outline}\n")

    response = await agent.run(f"Expand this outline into a short paragraph:\n\n{outline}")
    article = response.content
    print(f"Article:\n{article}\n")

    response = await agent.run(f"Write a title for this article:\n\n{article}")
    print(f"Title: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
