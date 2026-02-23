import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def main():
    agent = Agent(api_key=API_KEY, model="claude-sonnet-4-20250514")

    response = await agent.run("Write a short tweet about AI.")
    draft = response.content
    print(f"Draft: {draft}\n")

    response = await agent.run(f"Critique this tweet and suggest improvements:\n{draft}")
    critique = response.content
    print(f"Critique: {critique}\n")

    response = await agent.run(
        f"Rewrite this tweet based on the feedback:\n\nOriginal: {draft}\nFeedback: {critique}"
    )
    print(f"Final: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
