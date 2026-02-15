"""
Evaluator-Optimizer pattern.

Generate output, evaluate it, refine based on feedback.
"""

from barebone import complete, user


def evaluate_and_optimize(task: str, max_iterations: int = 3) -> str:
    """Generate, evaluate, and refine until good enough."""
    print("=" * 60)
    print("Evaluator-Optimizer")
    print("=" * 60)

    # Initial generation
    response = complete("claude-sonnet-4-20250514", [
        user(task)
    ])
    output = response.content
    print(f"Initial output:\n{output}\n")

    for i in range(max_iterations):
        # Evaluate
        evaluation = complete("claude-sonnet-4-20250514", [
            user(f"""Evaluate this output for the task: "{task}"

Output:
{output}

Rate 1-10 and list specific improvements needed.
If score >= 8, just respond "GOOD".
""")
        ]).content

        print(f"Iteration {i + 1} evaluation:\n{evaluation}\n")

        if "GOOD" in evaluation.upper():
            print("Output passed evaluation!")
            break

        # Optimize based on feedback
        response = complete("claude-sonnet-4-20250514", [
            user(f"""Improve this output based on feedback.

Original task: {task}

Current output:
{output}

Feedback:
{evaluation}

Provide improved output:""")
        ])
        output = response.content
        print(f"Improved output:\n{output}\n")

    return output


def critic_loop(task: str) -> str:
    """Separate generator and critic models."""
    print("\n" + "=" * 60)
    print("Generator-Critic Loop")
    print("=" * 60)

    # Generator creates
    output = complete("claude-sonnet-4-20250514", [
        user(task)
    ], temperature=0.9).content

    print(f"Generator:\n{output}\n")

    # Critic reviews
    critique = complete("claude-sonnet-4-20250514", [
        user(f"""You are a harsh critic. Find 2 specific problems with this:

{output}

Be specific and constructive.""")
    ], temperature=0.3).content

    print(f"Critic:\n{critique}\n")

    # Generator revises
    final = complete("claude-sonnet-4-20250514", [
        user(f"""Revise your work based on this critique:

Original: {output}

Critique: {critique}

Revised version:""")
    ]).content

    print(f"Revised:\n{final}")
    return final


if __name__ == "__main__":
    evaluate_and_optimize("Write a haiku about coding that uses a clever metaphor")
    critic_loop("Write a one-paragraph product description for noise-canceling headphones")
