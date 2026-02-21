"""Orchestrator-worker examples - task decomposition and parallel execution."""

import asyncio
import json
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def orchestrator_worker(task: str) -> str:
    """Break down task, execute subtasks, synthesize results."""
    print("=" * 60)
    print("Orchestrator-Worker")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    orchestrator = Agent(provider=provider)

    # Break down task
    response = await orchestrator.run(f"""Break this task into 2-3 subtasks.
Return as JSON array of strings.

Task: {task}

Example output: ["subtask 1", "subtask 2"]""")

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [task]

    print(f"Orchestrator broke down into {len(subtasks)} subtasks:")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. {st}")

    # Execute subtasks sequentially
    results = []
    for subtask in subtasks:
        worker = Agent(provider=provider)
        result = (await worker.run(f"Complete this task concisely: {subtask}")).content
        results.append(result)
        print(f"\nWorker completed: {subtask[:50]}...")

    # Synthesize results
    combined = "\n\n".join(f"Subtask: {st}\nResult: {r}" for st, r in zip(subtasks, results))
    response = await orchestrator.run(
        f"Synthesize these results into a final response:\n\n{combined}"
    )

    return response.content


async def async_orchestrator(task: str) -> str:
    """Parallel execution of independent subtasks."""
    print("\n" + "=" * 60)
    print("Async Orchestrator (Parallel Workers)")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    orchestrator = Agent(provider=provider)

    # Break down task
    response = await orchestrator.run(f"""Break this task into 2-3 independent subtasks.
Return as JSON array of strings.

Task: {task}""")

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [task]

    print(f"Subtasks: {subtasks}")

    # Execute subtasks in parallel
    async def execute_subtask(subtask: str) -> str:
        worker = Agent(provider=provider)
        response = await worker.run(f"Complete this task concisely: {subtask}")
        return response.content

    results = await asyncio.gather(*[execute_subtask(st) for st in subtasks])

    # Synthesize results
    combined = "\n\n".join(f"- {r}" for r in results)
    response = await orchestrator.run(f"Combine these into a coherent response:\n{combined}")

    return response.content


async def main():
    result = await orchestrator_worker("Write a short blog post about Python best practices")
    print(f"\nFinal Result:\n{result}")

    result = await async_orchestrator("Compare Python, Rust, and Go for web development")
    print(f"\nFinal Result:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
