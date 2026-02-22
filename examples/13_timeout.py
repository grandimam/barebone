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
        timeout=30.0,
    )

    try:
        response = await agent.run("Write a short story", timeout=10.0)
        print(response.content)
    except asyncio.TimeoutError:
        print("Request timed out")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
