"""Basic bare example - 5 lines to a working agent."""

from barebone import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


def main():
    # Create agent - credentials auto-discovered
    agent = Agent("claude-sonnet-4-20250514", tools=[get_weather, calculate])

    # Run queries
    print(agent.run_sync("What's the weather in Tokyo?").content)
    print(agent.run_sync("What is 123 * 456?").content)


if __name__ == "__main__":
    main()
