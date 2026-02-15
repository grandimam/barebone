"""
Orchestrator-Worker pattern.

One LLM breaks down tasks and delegates to worker LLMs.
"""

import asyncio
import json

from barebone import acomplete, complete, user


def orchestrator_worker(task: str) -> str:
    """Orchestrator breaks down task, workers execute subtasks."""
    print("=" * 60)
    print("Orchestrator-Worker")
    print("=" * 60)

    # Orchestrator: Break down the task
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Break this task into 2-3 subtasks.
Return as JSON array of strings.

Task: {task}

Example output: ["subtask 1", "subtask 2"]""")
    ])

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [task]

    print(f"Orchestrator broke down into {len(subtasks)} subtasks:")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. {st}")

    # Workers: Execute each subtask
    results = []
    for subtask in subtasks:
        result = complete("claude-sonnet-4-20250514", [
            user(f"Complete this task concisely: {subtask}")
        ]).content
        results.append(result)
        print(f"\nWorker completed: {subtask[:50]}...")

    # Orchestrator: Synthesize results
    combined = "\n\n".join(f"Subtask: {st}\nResult: {r}" for st, r in zip(subtasks, results))
    response = complete("claude-sonnet-4-20250514", [
        user(f"Synthesize these results into a final response:\n\n{combined}")
    ])

    return response.content


async def async_orchestrator(task: str) -> str:
    """Async version with parallel workers."""
    print("\n" + "=" * 60)
    print("Async Orchestrator (Parallel Workers)")
    print("=" * 60)

    # Orchestrator
    response = await acomplete("claude-sonnet-4-20250514", [
        user(f"""Break this task into 2-3 independent subtasks.
Return as JSON array of strings.

Task: {task}""")
    ])

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [task]

    print(f"Subtasks: {subtasks}")

    # Parallel workers
    tasks = [
        acomplete("claude-sonnet-4-20250514", [
            user(f"Complete this task concisely: {st}")
        ])
        for st in subtasks
    ]
    responses = await asyncio.gather(*tasks)
    results = [r.content for r in responses]

    # Synthesize
    combined = "\n\n".join(f"- {r}" for r in results)
    response = await acomplete("claude-sonnet-4-20250514", [
        user(f"Combine these into a coherent response:\n{combined}")
    ])

    return response.content


if __name__ == "__main__":
    result = orchestrator_worker("Write a short blog post about Python best practices")
    print(f"\nFinal Result:\n{result}")

    result = asyncio.run(async_orchestrator("Compare Python, Rust, and Go for web development"))
    print(f"\nFinal Result:\n{result}")
