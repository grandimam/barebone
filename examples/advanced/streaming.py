import asyncio
import os

from barebone.common.dataclasses import Done
from barebone.common.dataclasses import TextDelta

from barebone import astream
from barebone import user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


async def basic_streaming():
    print("=" * 60)
    print("Basic Streaming")
    print("=" * 60)

    async for event in astream(MODEL, [user("Write a haiku about Python.")], api_key=API_KEY):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Done):
            print(f"\n\nTokens: {event.response.usage.total_tokens}")


async def streaming_with_accumulation():
    print("\n" + "=" * 60)
    print("Streaming with Accumulation")
    print("=" * 60)

    full_content = ""

    async for event in astream(MODEL, [user("List 3 programming languages.")], api_key=API_KEY):
        if isinstance(event, TextDelta):
            full_content += event.text
            print(event.text, end="", flush=True)

    print(f"\n\nAccumulated {len(full_content)} characters")


async def streaming_with_processing():
    print("\n" + "=" * 60)
    print("Streaming with Real-time Processing")
    print("=" * 60)

    word_count = 0
    buffer = ""

    async for event in astream(MODEL, [user("Explain recursion briefly.")], api_key=API_KEY):
        if isinstance(event, TextDelta):
            buffer += event.text

            while " " in buffer:
                word, buffer = buffer.split(" ", 1)
                if word:
                    word_count += 1

            print(event.text, end="", flush=True)

    if buffer.strip():
        word_count += 1

    print(f"\n\nWord count: {word_count}")


async def parallel_streaming():
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
        async for event in astream(MODEL, [user(prompt)], api_key=API_KEY):
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
