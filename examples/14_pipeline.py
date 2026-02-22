import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def main():
    researcher = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are a researcher. Output only facts and data points.",
    )

    writer = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are a writer. Create engaging content from the facts provided.",
    )

    editor = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are an editor. Improve clarity and fix errors. Output the final version.",
    )

    topic = "The impact of AI on healthcare"

    print("Step 1: Research")
    research = await researcher.run(f"Research key facts about: {topic}")
    print(f"Research: {research.content[:200]}...\n")

    print("Step 2: Write")
    draft = await writer.run(f"Write a short article using these facts:\n\n{research.content}")
    print(f"Draft: {draft.content[:200]}...\n")

    print("Step 3: Edit")
    final = await editor.run(f"Edit and improve this article:\n\n{draft.content}")
    print(f"Final: {final.content}")


if __name__ == "__main__":
    asyncio.run(main())
