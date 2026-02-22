import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


async def route(query: str) -> str:
    router = Agent(api_key=API_KEY, model="claude-sonnet-4-20250514")

    response = await router.run(f"""Classify this query into one category.
Reply with just the category name.

Categories: tech, billing, general

Query: {query}""")

    category = response.content.strip().lower()
    print(f"  Routed to: {category}")

    if category == "tech":
        agent = Agent(
            api_key=API_KEY,
            model="claude-sonnet-4-20250514",
            system="You are a technical support expert. Be concise.",
        )
    elif category == "billing":
        agent = Agent(
            api_key=API_KEY,
            model="claude-sonnet-4-20250514",
            system="You are a billing specialist. Be helpful.",
        )
    else:
        agent = Agent(
            api_key=API_KEY,
            model="claude-sonnet-4-20250514",
            system="You are a helpful assistant.",
        )

    response = await agent.run(query)
    return response.content


async def main():
    queries = [
        "My internet keeps disconnecting",
        "How do I update my credit card?",
        "What's the weather like today?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        response = await route(q)
        print(f"Response: {response[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
