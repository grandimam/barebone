import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def main():
    technical = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are a technical analyst. Evaluate from engineering perspective.",
    )

    business = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are a business analyst. Evaluate from market perspective.",
    )

    user = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are a UX researcher. Evaluate from user perspective.",
    )

    synthesizer = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You synthesize multiple viewpoints into a balanced recommendation.",
    )

    idea = "A mobile app that uses AI to help people learn languages through conversations"

    print("Analyzing in parallel...\n")

    technical_task = technical.run(f"Analyze this idea: {idea}")
    business_task = business.run(f"Analyze this idea: {idea}")
    user_task = user.run(f"Analyze this idea: {idea}")

    results = await asyncio.gather(technical_task, business_task, user_task)

    print("Technical View:")
    print(f"{results[0].content[:200]}...\n")

    print("Business View:")
    print(f"{results[1].content[:200]}...\n")

    print("User View:")
    print(f"{results[2].content[:200]}...\n")

    combined = "\n\n".join(
        [
            f"Technical: {results[0].content}",
            f"Business: {results[1].content}",
            f"User: {results[2].content}",
        ]
    )

    print("Synthesizing...\n")
    final = await synthesizer.run(f"Synthesize these perspectives:\n\n{combined}")
    print(f"Final Recommendation:\n{final.content}")


if __name__ == "__main__":
    asyncio.run(main())
