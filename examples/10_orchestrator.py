import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def worker(task: str, role: str) -> str:
    agent = Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
        system=f"You are a {role}. Complete the task concisely.",
    )
    response = await agent.run(task)
    return response.content


async def main():
    orchestrator = Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
        system="You are a project manager. Break down tasks and assign them.",
    )

    response = await orchestrator.run(
        "Create a marketing plan. List 3 tasks with roles: writer, designer, analyst"
    )
    print(f"Plan:\n{response.content}\n")

    # Simulate worker execution
    tasks = [
        ("Write a tagline for AI product", "copywriter"),
        ("Suggest color scheme", "designer"),
        ("List target demographics", "analyst"),
    ]

    results = await asyncio.gather(*[worker(t, r) for t, r in tasks])

    print("Worker Results:")
    for (task, role), result in zip(tasks, results):
        print(f"  [{role}] {result[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
