import os
import time
from collections.abc import Callable

from barebone import complete
from barebone import user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


def with_retry(fn: Callable, max_retries: int = 3, backoff: float = 1.0) -> any:
    last_error = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            wait = backoff * (2**attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise last_error


def retry_with_modification(task: str, max_retries: int = 3) -> str:
    print("=" * 60)
    print("Retry with Modification")
    print("=" * 60)

    errors = []

    for attempt in range(max_retries):
        if errors:
            error_context = "\n\nPrevious attempts failed:\n" + "\n".join(f"- {e}" for e in errors)
            error_context += "\n\nAvoid these issues."
        else:
            error_context = ""

        response = complete(MODEL, [user(f"{task}{error_context}")], api_key=API_KEY)
        result = response.content

        if "error" in result.lower() or len(result) < 10:
            errors.append(f"Attempt {attempt + 1}: Output too short or contained error")
            print(f"Attempt {attempt + 1} failed, retrying...")
            continue

        print(f"Success on attempt {attempt + 1}")
        return result

    return f"Failed after {max_retries} attempts. Last errors: {errors}"


def fallback_chain(task: str) -> str:
    print("\n" + "=" * 60)
    print("Fallback Chain")
    print("=" * 60)

    strategies = [
        ("Direct", lambda t: complete(MODEL, [user(t)], api_key=API_KEY).content),
        (
            "Step-by-step",
            lambda t: complete(MODEL, [user(f"Think step by step: {t}")], api_key=API_KEY).content,
        ),
        (
            "Simple",
            lambda t: complete(
                MODEL, [user(f"Give a simple, basic answer: {t}")], api_key=API_KEY
            ).content,
        ),
    ]

    for name, strategy in strategies:
        try:
            print(f"Trying {name} strategy...")
            result = strategy(task)

            if result and len(result) > 20:
                print(f"Success with {name} strategy")
                return result

            print(f"{name} produced insufficient result")
        except Exception as e:
            print(f"{name} failed: {e}")

    return "All strategies failed"


def model_fallback(task: str) -> str:
    print("\n" + "=" * 60)
    print("Model Fallback")
    print("=" * 60)

    models = [MODEL, MODEL]

    for model in models:
        try:
            print(f"Trying {model}...")
            response = complete(model, [user(task)], api_key=API_KEY)
            print(f"Success with {model}")
            return response.content
        except Exception as e:
            print(f"{model} failed: {e}")

    return "All models failed"


def self_healing(task: str, max_attempts: int = 3) -> str:
    print("\n" + "=" * 60)
    print("Self-Healing")
    print("=" * 60)

    response = complete(MODEL, [user(task)], api_key=API_KEY)
    result = response.content

    for attempt in range(max_attempts):
        validation = complete(
            MODEL,
            [
                user(f"""Check if this response correctly addresses the task.
If there are errors or issues, describe them.
If it's correct, respond with "VALID".

Task: {task}

Response: {result}""")
            ],
            api_key=API_KEY,
        ).content

        print(f"Validation {attempt + 1}: {validation[:100]}...")

        if "VALID" in validation.upper():
            print("Response validated successfully")
            return result

        result = complete(
            MODEL,
            [
                user(f"""Fix the issues identified:

Task: {task}

Previous response: {result}

Issues: {validation}

Fixed response:""")
            ],
            api_key=API_KEY,
        ).content

        print(f"Self-healed attempt {attempt + 1}")

    return result


if __name__ == "__main__":
    retry_with_modification("Write a haiku about programming")
    fallback_chain("Explain quantum entanglement simply")
    model_fallback("What is 2 + 2?")
    self_healing("List the first 5 prime numbers")
