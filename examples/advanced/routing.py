"""
Routing pattern.

Classify input and route to specialized handlers.
"""

from barebone import complete, user


def route(query: str) -> str:
    """Classify and route query to appropriate handler."""

    # Step 1: Classify
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Classify this query into exactly one category.
Reply with just the category name.

Categories: tech_support, billing, general

Query: {query}""")
    ])
    category = response.content.strip().lower()

    # Step 2: Route to handler
    handlers = {
        "tech_support": handle_tech,
        "billing": handle_billing,
        "general": handle_general,
    }
    handler = handlers.get(category, handle_general)
    return handler(query)


def handle_tech(query: str) -> str:
    return complete("claude-sonnet-4-20250514", [
        user(f"You are a technical support expert. Help with: {query}")
    ], system="Be concise and technical. Provide step-by-step solutions.").content


def handle_billing(query: str) -> str:
    return complete("claude-sonnet-4-20250514", [
        user(f"You are a billing specialist. Help with: {query}")
    ], system="Be helpful and clear about pricing and payments.").content


def handle_general(query: str) -> str:
    return complete("claude-sonnet-4-20250514", [
        user(query)
    ], system="Be helpful and friendly.").content


if __name__ == "__main__":
    queries = [
        "My internet keeps disconnecting",
        "How do I update my credit card?",
        "What's the weather like today?",
    ]

    for q in queries:
        print(f"Query: {q}")
        print(f"Response: {route(q)}\n")
