import os

from barebone import complete, user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


def route(query: str) -> str:
    response = complete(MODEL, [
        user(f"""Classify this query into exactly one category.
Reply with just the category name.

Categories: tech_support, billing, general

Query: {query}""")
    ], api_key=API_KEY)
    category = response.content.strip().lower()

    handlers = {
        "tech_support": handle_tech,
        "billing": handle_billing,
        "general": handle_general,
    }
    handler = handlers.get(category, handle_general)
    return handler(query)


def handle_tech(query: str) -> str:
    return complete(MODEL, [
        user(f"You are a technical support expert. Help with: {query}")
    ], api_key=API_KEY, system="Be concise and technical. Provide step-by-step solutions.").content


def handle_billing(query: str) -> str:
    return complete(MODEL, [
        user(f"You are a billing specialist. Help with: {query}")
    ], api_key=API_KEY, system="Be helpful and clear about pricing and payments.").content


def handle_general(query: str) -> str:
    return complete(MODEL, [
        user(query)
    ], api_key=API_KEY, system="Be helpful and friendly.").content


if __name__ == "__main__":
    queries = [
        "My internet keeps disconnecting",
        "How do I update my credit card?",
        "What's the weather like today?",
    ]

    for q in queries:
        print(f"Query: {q}")
        print(f"Response: {route(q)}\n")
