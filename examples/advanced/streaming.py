"""
Streaming pattern.

Stream LLM responses for real-time output.
"""

import asyncio

from barebone import astream, user, TextDelta, Done


async def basic_streaming():
    """Basic streaming output."""
    print("=" * 60)
    print("Basic Streaming")
    print("=" * 60)

    async for event in astream("claude-sonnet-4-20250514", [user("Write a haiku about Python.")]):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Done):
            print(f"\n\nTokens: {event.response.usage.total_tokens}")


async def streaming_with_accumulation():
    """Accumulate streamed content."""
    print("\n" + "=" * 60)
    print("Streaming with Accumulation")
    print("=" * 60)

    full_content = ""

    async for event in astream("claude-sonnet-4-20250514", [user("List 3 programming languages.")]):
        if isinstance(event, TextDelta):
            full_content += event.text
            print(event.text, end="", flush=True)

    print(f"\n\nAccumulated {len(full_content)} characters")


async def streaming_with_processing():
    """Process streamed content in real-time."""
    print("\n" + "=" * 60)
    print("Streaming with Real-time Processing")
    print("=" * 60)

    word_count = 0
    buffer = ""

    async for event in astream("claude-sonnet-4-20250514", [user("Explain recursion briefly.")]):
        if isinstance(event, TextDelta):
            buffer += event.text

            # Count complete words
            while " " in buffer:
                word, buffer = buffer.split(" ", 1)
                if word:
                    word_count += 1

            print(event.text, end="", flush=True)

    # Count remaining word in buffer
    if buffer.strip():
        word_count += 1

    print(f"\n\nWord count: {word_count}")


async def parallel_streaming():
    """Stream multiple responses in parallel."""
    print("\n" + "=" * 60)
    print("Parallel Streaming")
    print("=" * 60)

    prompts = [
        "Say 'Hello' in French",
        "Say 'Hello' in Spanish",
        "Say 'Hello' in Japanese",
    ]

    async def stream_one(prompt: str, label: str):
        content = ""
        async for event in astream("claude-sonnet-4-20250514", [user(prompt)]):
            if isinstance(event, TextDelta):
                content += event.text
        return label, content

    tasks = [stream_one(p, p.split("'")[2].strip()) for p in prompts]
    results = await asyncio.gather(*tasks)

    for label, content in results:
        print(f"{label}: {content}")


if __name__ == "__main__":
    asyncio.run(basic_streaming())
    asyncio.run(streaming_with_accumulation())
    asyncio.run(streaming_with_processing())
    asyncio.run(parallel_streaming())
