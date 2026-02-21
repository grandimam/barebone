"""Evaluator-optimizer examples - iterative improvement with evaluation."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def evaluate_and_optimize(task: str, max_iterations: int = 3) -> str:
    """Iterative optimization with evaluation scoring."""
    print("=" * 60)
    print("Evaluator-Optimizer")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    response = await agent.run(task)
    output = response.content
    print(f"Initial output:\n{output}\n")

    for i in range(max_iterations):
        evaluation = (await agent.run(f"""Evaluate this output for the task: "{task}"

Output:
{output}

Rate 1-10 and list specific improvements needed.
If score >= 8, just respond "GOOD".
""")).content

        print(f"Iteration {i + 1} evaluation:\n{evaluation}\n")

        if "GOOD" in evaluation.upper():
            print("Output passed evaluation!")
            break

        output = (await agent.run(f"""Improve this output based on feedback.

Original task: {task}

Current output:
{output}

Feedback:
{evaluation}

Provide improved output:""")).content

        print(f"Improved output:\n{output}\n")

    return output


async def critic_loop(task: str) -> str:
    """Generator-critic pattern with harsh feedback."""
    print("\n" + "=" * 60)
    print("Generator-Critic Loop")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    # Generator with high temperature
    generator = Agent(provider=provider, temperature=0.9)
    output = (await generator.run(task)).content
    print(f"Generator:\n{output}\n")

    # Critic with low temperature
    critic = Agent(provider=provider, temperature=0.3)
    critique = (await critic.run(f"""You are a harsh critic. Find 2 specific problems with this:

{output}

Be specific and constructive.""")).content
    print(f"Critic:\n{critique}\n")

    # Final revision
    final = (await generator.run(f"""Revise your work based on this critique:

Original: {output}

Critique: {critique}

Revised version:""")).content

    print(f"Revised:\n{final}")
    return final


async def main():
    await evaluate_and_optimize("Write a haiku about coding that uses a clever metaphor")
    await critic_loop("Write a one-paragraph product description for noise-canceling headphones")


if __name__ == "__main__":
    asyncio.run(main())
