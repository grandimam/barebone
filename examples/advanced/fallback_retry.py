"""Fallback and retry examples - error handling patterns."""

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def with_retry(fn: Callable, max_retries: int = 3, backoff: float = 1.0) -> Any:
    """Retry wrapper with exponential backoff."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            wait = backoff * (2**attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    raise last_error


async def retry_with_modification(task: str, max_retries: int = 3) -> str:
    """Retry with modified prompts based on previous failures."""
    print("=" * 60)
    print("Retry with Modification")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)
    errors = []

    for attempt in range(max_retries):
        if errors:
            error_context = "\n\nPrevious attempts failed:\n" + "\n".join(f"- {e}" for e in errors)
            error_context += "\n\nAvoid these issues."
        else:
            error_context = ""

        response = await agent.run(f"{task}{error_context}")
        result = response.content

        if "error" in result.lower() or len(result) < 10:
            errors.append(f"Attempt {attempt + 1}: Output too short or contained error")
            print(f"Attempt {attempt + 1} failed, retrying...")
            continue

        print(f"Success on attempt {attempt + 1}")
        return result

    return f"Failed after {max_retries} attempts. Last errors: {errors}"


async def fallback_chain(task: str) -> str:
    """Try multiple strategies in sequence until one succeeds."""
    print("\n" + "=" * 60)
    print("Fallback Chain")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    strategies = [
        ("Direct", task),
        ("Step-by-step", f"Think step by step: {task}"),
        ("Simple", f"Give a simple, basic answer: {task}"),
    ]

    for name, prompt in strategies:
        try:
            print(f"Trying {name} strategy...")
            agent = Agent(provider=provider)
            response = await agent.run(prompt)
            result = response.content

            if result and len(result) > 20:
                print(f"Success with {name} strategy")
                return result

            print(f"{name} produced insufficient result")
        except Exception as e:
            print(f"{name} failed: {e}")

    return "All strategies failed"


async def model_fallback(task: str) -> str:
    """Try different models in sequence."""
    print("\n" + "=" * 60)
    print("Model Fallback")
    print("=" * 60)

    models = ["claude-sonnet-4-20250514", "claude-sonnet-4-20250514"]

    for model in models:
        try:
            print(f"Trying {model}...")
            provider = AnthropicProvider(api_key=API_KEY, model=model)
            agent = Agent(provider=provider)
            response = await agent.run(task)
            print(f"Success with {model}")
            return response.content
        except Exception as e:
            print(f"{model} failed: {e}")

    return "All models failed"


async def self_healing(task: str, max_attempts: int = 3) -> str:
    """Self-healing pattern: validate and fix own output."""
    print("\n" + "=" * 60)
    print("Self-Healing")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    response = await agent.run(task)
    result = response.content

    for attempt in range(max_attempts):
        validation = (await agent.run(f"""Check if this response correctly addresses the task.
If there are errors or issues, describe them.
If it's correct, respond with "VALID".

Task: {task}

Response: {result}""")).content

        print(f"Validation {attempt + 1}: {validation[:100]}...")

        if "VALID" in validation.upper():
            print("Response validated successfully")
            return result

        result = (await agent.run(f"""Fix the issues identified:

Task: {task}

Previous response: {result}

Issues: {validation}

Fixed response:""")).content

        print(f"Self-healed attempt {attempt + 1}")

    return result


async def main():
    await retry_with_modification("Write a haiku about programming")
    await fallback_chain("Explain quantum entanglement simply")
    await model_fallback("What is 2 + 2?")
    await self_healing("List the first 5 prime numbers")


if __name__ == "__main__":
    asyncio.run(main())
