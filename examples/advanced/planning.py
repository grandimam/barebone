"""
Planning pattern.

Generate a plan first, then execute each step.
"""

import json

from barebone import complete, user, Tool, Param, execute


class SearchWeb(Tool):
    """Search the web for information."""
    query: str = Param(description="Search query")

    def execute(self) -> str:
        return f"[Search results for '{self.query}': Found 3 relevant articles]"


class WriteFile(Tool):
    """Write content to a file."""
    filename: str = Param(description="File name")
    content: str = Param(description="Content to write")

    def execute(self) -> str:
        return f"[Wrote {len(self.content)} chars to {self.filename}]"


class ReadFile(Tool):
    """Read a file."""
    filename: str = Param(description="File name")

    def execute(self) -> str:
        return f"[Content of {self.filename}: Example content here]"


def plan_and_execute(task: str, tools: list) -> str:
    """Generate plan, then execute step by step."""
    print("=" * 60)
    print("Plan and Execute")
    print("=" * 60)

    tool_names = [t.get_name() for t in tools]

    # Step 1: Generate plan
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Create a step-by-step plan for this task.
Available tools: {tool_names}

Task: {task}

Return as JSON array:
[
  {{"step": 1, "action": "description", "tool": "tool_name or null"}},
  ...
]""")
    ])

    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        plan = [{"step": 1, "action": task, "tool": None}]

    print("Generated Plan:")
    for step in plan:
        print(f"  {step['step']}. {step['action']} (tool: {step.get('tool', 'none')})")

    # Step 2: Execute each step
    print("\nExecuting Plan:")
    results = []

    for step in plan:
        print(f"\n--- Step {step['step']}: {step['action']} ---")

        if step.get("tool"):
            # Let LLM decide tool parameters
            response = complete("claude-sonnet-4-20250514", [
                user(f"Execute: {step['action']}")
            ], tools=tools)

            if response.tool_calls:
                for tc in response.tool_calls:
                    result = execute(tc, tools)
                    print(f"  {tc.name}: {result}")
                    results.append(result)
            else:
                results.append(response.content)
        else:
            # No tool needed, just reason
            response = complete("claude-sonnet-4-20250514", [
                user(f"Complete this step: {step['action']}\n\nContext from previous steps: {results[-3:] if results else 'None'}")
            ])
            results.append(response.content)
            print(f"  Result: {response.content[:100]}...")

    # Step 3: Summarize
    response = complete("claude-sonnet-4-20250514", [
        user(f"Summarize what was accomplished:\n\nTask: {task}\n\nResults: {results}")
    ])

    return response.content


def adaptive_planning(task: str) -> str:
    """Replan based on execution results."""
    print("\n" + "=" * 60)
    print("Adaptive Planning (Replan on Failure)")
    print("=" * 60)

    max_replans = 2
    context = []

    for attempt in range(max_replans + 1):
        # Generate/regenerate plan
        context_str = "\n".join(context) if context else "No previous attempts"

        response = complete("claude-sonnet-4-20250514", [
            user(f"""Task: {task}

Previous attempts and issues:
{context_str}

Create a plan that avoids previous issues. Return 2-3 concrete steps.""")
        ])
        plan = response.content
        print(f"\nPlan (attempt {attempt + 1}):\n{plan}")

        # Simulate execution (in real scenario, actually execute)
        response = complete("claude-sonnet-4-20250514", [
            user(f"""Simulate executing this plan.
If any step would fail, explain why.
If all steps succeed, respond with "SUCCESS" and the result.

Plan:
{plan}""")
        ])
        result = response.content

        if "SUCCESS" in result.upper():
            print(f"\nSuccess: {result}")
            return result

        print(f"\nIssue: {result}")
        context.append(f"Attempt {attempt + 1}: {plan}\nFailed because: {result}")

    return "Max replans reached"


if __name__ == "__main__":
    tools = [SearchWeb, WriteFile, ReadFile]
    result = plan_and_execute("Research Python async patterns and save a summary to notes.txt", tools)
    print(f"\nFinal Summary:\n{result}")

    adaptive_planning("Find a way to make coffee without a coffee maker")
