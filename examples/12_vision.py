import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def main():
    async with Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
    ) as agent:
        response = await agent.run(
            "What do you see in this image?",
            images=[
                "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
            ],
        )
        print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
