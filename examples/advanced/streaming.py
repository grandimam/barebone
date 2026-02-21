"""Streaming examples demonstrating real-time response handling."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import AnthropicProvider
from barebone.types import Message

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def basic_streaming():
    """Basic streaming with text output."""
    print("=" * 60)
    print("Basic Streaming")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    messages = [Message(role="user", content="Write a haiku about Python.")]

    async for event in provider.stream(messages=messages):
        if event.get("type") == "text_delta":
            print(event.get("text", ""), end="", flush=True)
        elif event.get("type") == "done":
            print("\n")


async def streaming_with_accumulation():
    """Streaming while accumulating the full response."""
    print("=" * 60)
    print("Streaming with Accumulation")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    messages = [Message(role="user", content="List 3 programming languages.")]

    full_content = ""
    async for event in provider.stream(messages=messages):
        if event.get("type") == "text_delta":
            text = event.get("text", "")
            full_content += text
            print(text, end="", flush=True)

    print(f"\n\nAccumulated {len(full_content)} characters")


async def streaming_with_processing():
    """Streaming with real-time word counting."""
    print("\n" + "=" * 60)
    print("Streaming with Real-time Processing")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    messages = [Message(role="user", content="Explain recursion briefly.")]

    word_count = 0
    buffer = ""

    async for event in provider.stream(messages=messages):
        if event.get("type") == "text_delta":
            text = event.get("text", "")
            buffer += text

            # Count complete words
            while " " in buffer:
                word, buffer = buffer.split(" ", 1)
                if word:
                    word_count += 1

            print(text, end="", flush=True)

    # Count any remaining word
    if buffer.strip():
        word_count += 1

    print(f"\n\nWord count: {word_count}")


async def parallel_streaming():
    """Multiple streams running in parallel."""
    print("\n" + "=" * 60)
    print("Parallel Streaming")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    prompts = [
        ("French", "Say 'Hello' in French"),
        ("Spanish", "Say 'Hello' in Spanish"),
        ("Japanese", "Say 'Hello' in Japanese"),
    ]

    async def stream_one(label: str, prompt: str) -> tuple[str, str]:
        messages = [Message(role="user", content=prompt)]
        content = ""
        async for event in provider.stream(messages=messages):
            if event.get("type") == "text_delta":
                content += event.get("text", "")
        return label, content

    tasks = [stream_one(label, prompt) for label, prompt in prompts]
    results = await asyncio.gather(*tasks)

    for label, content in results:
        print(f"{label}: {content}")


async def main():
    await basic_streaming()
    await streaming_with_accumulation()
    await streaming_with_processing()
    await parallel_streaming()


if __name__ == "__main__":
    asyncio.run(main())
