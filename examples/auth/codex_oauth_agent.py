import asyncio
from barebone import Agent


async def simple_chat():
    agent = Agent(
        model="gpt-5-codex",
        provider="openai-codex",
    )

    response = await agent.run("What is the capital of France?")
    print(f"Response: {response.content}")


async def with_system_prompt():
    agent = Agent(
        model="gpt-5-codex",
        provider="openai-codex",
        system="You are a helpful coding assistant. Be concise.",
    )

    response = await agent.run("Write a Python function to check if a number is prime.")
    print(f"Response:\n{response.content}")


async def with_tools():
    from barebone import tool

    @tool
    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny, 22°C"

    agent = Agent(
        model="gpt-5-codex",
        provider="openai-codex",
        tools=[get_weather],
    )

    response = await agent.run("What's the weather in Tokyo?")
    print(f"Response: {response.content}")


async def multi_turn():
    agent = Agent(
        model="gpt-5-codex",
        provider="openai-codex",
    )

    response = await agent.run("My name is Alice.")
    print(f"Turn 1: {response.content}")

    # Second turn - agent remembers context
    response = await agent.run("What is my name?")
    print(f"Turn 2: {response.content}")


if __name__ == "__main__":
    print("=" * 60)
    print("Barebone Agent with OpenAI Codex")
    print("=" * 60)

    print("\n[1] Simple chat")
    asyncio.run(simple_chat())

    print("\n[2] With system prompt")
    asyncio.run(with_system_prompt())

    print("\n[3] With tools")
    asyncio.run(with_tools())

    print("\n[5] Multi-turn conversation")
    asyncio.run(multi_turn())
