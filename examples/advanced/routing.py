"""Query routing example - directing queries to specialized handlers."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


async def route(query: str) -> str:
    """Route a query to the appropriate handler based on classification."""
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    router = Agent(provider=provider)

    response = await router.run(f"""Classify this query into exactly one category.
Reply with just the category name.

Categories: tech_support, billing, general

Query: {query}""")

    category = response.content.strip().lower()
    print(f"  Routed to: {category}")

    handlers = {
        "tech_support": handle_tech,
        "billing": handle_billing,
        "general": handle_general,
    }
    handler = handlers.get(category, handle_general)
    return await handler(query)


async def handle_tech(query: str) -> str:
    """Technical support handler."""
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="Be concise and technical. Provide step-by-step solutions.",
    )
    response = await agent.run(f"You are a technical support expert. Help with: {query}")
    return response.content


async def handle_billing(query: str) -> str:
    """Billing specialist handler."""
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="Be helpful and clear about pricing and payments.",
    )
    response = await agent.run(f"You are a billing specialist. Help with: {query}")
    return response.content


async def handle_general(query: str) -> str:
    """General query handler."""
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(
        provider=provider,
        system="Be helpful and friendly.",
    )
    response = await agent.run(query)
    return response.content


async def main():
    print("=" * 60)
    print("Query Routing Example")
    print("=" * 60)

    queries = [
        "My internet keeps disconnecting",
        "How do I update my credit card?",
        "What's the weather like today?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        response = await route(q)
        print(f"Response: {response[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
