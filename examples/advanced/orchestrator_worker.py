import asyncio
import json
import os

from barebone import acomplete, complete, user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


def orchestrator_worker(task: str) -> str:
    print("=" * 60)
    print("Orchestrator-Worker")
    print("=" * 60)

    response = complete(MODEL, [
        user(f"""Break this task into 2-3 subtasks.
Return as JSON array of strings.

Task: {task}

Example output: ["subtask 1", "subtask 2"]""")
    ], api_key=API_KEY)

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [task]

    print(f"Orchestrator broke down into {len(subtasks)} subtasks:")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. {st}")

    results = []
    for subtask in subtasks:
        result = complete(MODEL, [
            user(f"Complete this task concisely: {subtask}")
        ], api_key=API_KEY).content
        results.append(result)
        print(f"\nWorker completed: {subtask[:50]}...")

    combined = "\n\n".join(f"Subtask: {st}\nResult: {r}" for st, r in zip(subtasks, results))
    response = complete(MODEL, [
        user(f"Synthesize these results into a final response:\n\n{combined}")
    ], api_key=API_KEY)

    return response.content


async def async_orchestrator(task: str) -> str:
    print("\n" + "=" * 60)
    print("Async Orchestrator (Parallel Workers)")
    print("=" * 60)

    response = await acomplete(MODEL, [
        user(f"""Break this task into 2-3 independent subtasks.
Return as JSON array of strings.

Task: {task}""")
    ], api_key=API_KEY)

    try:
        subtasks = json.loads(response.content)
    except json.JSONDecodeError:
        subtasks = [task]

    print(f"Subtasks: {subtasks}")

    tasks = [
        acomplete(MODEL, [user(f"Complete this task concisely: {st}")], api_key=API_KEY)
        for st in subtasks
    ]
    responses = await asyncio.gather(*tasks)
    results = [r.content for r in responses]

    combined = "\n\n".join(f"- {r}" for r in results)
    response = await acomplete(MODEL, [
        user(f"Combine these into a coherent response:\n{combined}")
    ], api_key=API_KEY)

    return response.content


if __name__ == "__main__":
    result = orchestrator_worker("Write a short blog post about Python best practices")
    print(f"\nFinal Result:\n{result}")

    result = asyncio.run(async_orchestrator("Compare Python, Rust, and Go for web development"))
    print(f"\nFinal Result:\n{result}")
