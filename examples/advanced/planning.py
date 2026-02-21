"""Planning examples - plan generation and execution patterns."""

import asyncio
import json
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider
from barebone import tool

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Search results for '{query}': Found 3 relevant articles]"


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    return f"[Wrote {len(content)} chars to {filename}]"


@tool
def read_file(filename: str) -> str:
    """Read a file."""
    return f"[Content of {filename}: Example content here]"


async def plan_and_execute(task: str) -> str:
    """Generate a plan and execute it step by step."""
    print("=" * 60)
    print("Plan and Execute")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    tools = [search_web, write_file, read_file]
    tool_names = [t.name for t in tools]

    # Generate plan
    planner = Agent(provider=provider)
    response = await planner.run(f"""Create a step-by-step plan for this task.
Available tools: {tool_names}

Task: {task}

Return as JSON array:
[
  {{"step": 1, "action": "description", "tool": "tool_name or null"}},
  ...
]""")

    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        plan = [{"step": 1, "action": task, "tool": None}]

    print("Generated Plan:")
    for step in plan:
        print(f"  {step['step']}. {step['action']} (tool: {step.get('tool', 'none')})")

    # Execute plan
    print("\nExecuting Plan:")
    executor = Agent(provider=provider, tools=tools)
    results = []

    for step in plan:
        print(f"\n--- Step {step['step']}: {step['action']} ---")

        if step.get("tool"):
            response = await executor.run(f"Execute: {step['action']}")
            results.append(response.content)
            print(f"  Result: {response.content[:100]}...")
        else:
            context = results[-3:] if results else []
            response = await executor.run(
                f"Complete this step: {step['action']}\n\nContext from previous steps: {context}"
            )
            results.append(response.content)
            print(f"  Result: {response.content[:100]}...")

    # Summarize
    response = await planner.run(
        f"Summarize what was accomplished:\n\nTask: {task}\n\nResults: {results}"
    )

    return response.content


async def adaptive_planning(task: str) -> str:
    """Planning with replanning on failure."""
    print("\n" + "=" * 60)
    print("Adaptive Planning (Replan on Failure)")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    max_replans = 2
    context = []

    for attempt in range(max_replans + 1):
        context_str = "\n".join(context) if context else "No previous attempts"

        response = await agent.run(f"""Task: {task}

Previous attempts and issues:
{context_str}

Create a plan that avoids previous issues. Return 2-3 concrete steps.""")
        plan = response.content
        print(f"\nPlan (attempt {attempt + 1}):\n{plan}")

        response = await agent.run(f"""Simulate executing this plan.
If any step would fail, explain why.
If all steps succeed, respond with "SUCCESS" and the result.

Plan:
{plan}""")
        result = response.content

        if "SUCCESS" in result.upper():
            print(f"\nSuccess: {result}")
            return result

        print(f"\nIssue: {result}")
        context.append(f"Attempt {attempt + 1}: {plan}\nFailed because: {result}")

    return "Max replans reached"


async def main():
    result = await plan_and_execute(
        "Research Python async patterns and save a summary to notes.txt"
    )
    print(f"\nFinal Summary:\n{result}")

    await adaptive_planning("Find a way to make coffee without a coffee maker")


if __name__ == "__main__":
    asyncio.run(main())
