"""Reflection examples - self-critique and improvement patterns."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def self_reflection(task: str, max_reflections: int = 2) -> str:
    """Self-reflection pattern: generate, reflect, improve."""
    print("=" * 60)
    print("Self Reflection")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    response = await agent.run(task)
    output = response.content
    print(f"Initial:\n{output}\n")

    for i in range(max_reflections):
        reflection = (await agent.run(f"""Reflect on your previous response to: "{task}"

Your response was:
{output}

What could be improved? Be specific. If it's already excellent, say "NO IMPROVEMENTS NEEDED".""")).content

        print(f"Reflection {i + 1}:\n{reflection}\n")

        if "NO IMPROVEMENTS NEEDED" in reflection.upper():
            break

        output = (await agent.run(f"""Improve your response based on your reflection.

Original task: {task}

Your previous response:
{output}

Your reflection:
{reflection}

Improved response:""")).content

        print(f"Improved:\n{output}\n")

    return output


async def chain_of_thought_reflection(question: str) -> str:
    """Chain-of-thought with verification."""
    print("\n" + "=" * 60)
    print("Chain-of-Thought Reflection")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    response = await agent.run(f"""Think through this step by step:

{question}

Show your reasoning, then give your answer.""")
    reasoning = response.content
    print(f"Initial reasoning:\n{reasoning}\n")

    reflection = (await agent.run(f"""Review your reasoning for any logical errors or gaps:

{reasoning}

List any issues with the reasoning process. If none, say "REASONING VALID".""")).content

    print(f"Reflection on reasoning:\n{reflection}\n")

    if "REASONING VALID" not in reflection.upper():
        response = await agent.run(f"""Redo your reasoning, addressing these issues:

{reflection}

Original question: {question}

Corrected reasoning and answer:""")
        reasoning = response.content
        print(f"Corrected reasoning:\n{reasoning}\n")

    return reasoning


async def debate_reflection(topic: str) -> str:
    """Debate pattern: argue both sides, then synthesize."""
    print("\n" + "=" * 60)
    print("Debate Reflection")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    position_a = (await agent.run(f"Argue strongly FOR: {topic}")).content
    print(f"For:\n{position_a}\n")

    position_b = (await agent.run(f"Argue strongly AGAINST: {topic}")).content
    print(f"Against:\n{position_b}\n")

    synthesis = (await agent.run(f"""Given these two positions, provide a balanced, nuanced view:

FOR: {position_a}

AGAINST: {position_b}

Synthesized view:""")).content

    print(f"Synthesis:\n{synthesis}")
    return synthesis


async def main():
    await self_reflection("Explain recursion to a beginner in 2-3 sentences")
    await chain_of_thought_reflection(
        "If a train leaves at 3pm going 60mph, and another leaves at 4pm going 80mph from 100 miles behind, when do they meet?"
    )
    await debate_reflection("AI will create more jobs than it eliminates")


if __name__ == "__main__":
    asyncio.run(main())
