import os

from barebone import complete, user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


def evaluate_and_optimize(task: str, max_iterations: int = 3) -> str:
    print("=" * 60)
    print("Evaluator-Optimizer")
    print("=" * 60)

    response = complete(MODEL, [user(task)], api_key=API_KEY)
    output = response.content
    print(f"Initial output:\n{output}\n")

    for i in range(max_iterations):
        evaluation = complete(MODEL, [
            user(f"""Evaluate this output for the task: "{task}"

Output:
{output}

Rate 1-10 and list specific improvements needed.
If score >= 8, just respond "GOOD".
""")
        ], api_key=API_KEY).content

        print(f"Iteration {i + 1} evaluation:\n{evaluation}\n")

        if "GOOD" in evaluation.upper():
            print("Output passed evaluation!")
            break

        response = complete(MODEL, [
            user(f"""Improve this output based on feedback.

Original task: {task}

Current output:
{output}

Feedback:
{evaluation}

Provide improved output:""")
        ], api_key=API_KEY)
        output = response.content
        print(f"Improved output:\n{output}\n")

    return output


def critic_loop(task: str) -> str:
    print("\n" + "=" * 60)
    print("Generator-Critic Loop")
    print("=" * 60)

    output = complete(MODEL, [user(task)], api_key=API_KEY, temperature=0.9).content
    print(f"Generator:\n{output}\n")

    critique = complete(MODEL, [
        user(f"""You are a harsh critic. Find 2 specific problems with this:

{output}

Be specific and constructive.""")
    ], api_key=API_KEY, temperature=0.3).content
    print(f"Critic:\n{critique}\n")

    final = complete(MODEL, [
        user(f"""Revise your work based on this critique:

Original: {output}

Critique: {critique}

Revised version:""")
    ], api_key=API_KEY).content

    print(f"Revised:\n{final}")
    return final


if __name__ == "__main__":
    evaluate_and_optimize("Write a haiku about coding that uses a clever metaphor")
    critic_loop("Write a one-paragraph product description for noise-canceling headphones")
