import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    return f"72°F and sunny in {city}"


@tool
def calculate(expression: str) -> str:
    return str(eval(expression))


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    agent = Agent("claude-sonnet-4-20250514", api_key=api_key, tools=[get_weather, calculate])
    print(agent.run_sync("What's the weather in Tokyo?").content)
    print(agent.run_sync("What is 123 * 456?").content)


if __name__ == "__main__":
    main()
