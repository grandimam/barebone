import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import tool

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@tool
def get_weather(city: str) -> str:
    return f"72F and sunny in {city}"


@tool
def calculate(expression: str) -> str:
    return str(eval(expression))


def main():
    agent = Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
        tools=[get_weather, calculate],
        system="You are a helpful assistant.",
    )

    response = agent.run_sync("What's the weather in Tokyo?")
    print(f"Weather: {response.content}")

    response = agent.run_sync("What is 123 * 456?")
    print(f"Math: {response.content}")


if __name__ == "__main__":
    main()
