from barebone import Agent
from barebone import tool
from barebone import CodexProvider
from barebone import load_credentials
from barebone import save_credentials


@tool
def get_weather(city: str) -> str:
    """Helps find the weather"""

    return f"72F and sunny in {city}"


@tool
def calculate(expression: str) -> str:
    """Helps calculate the expression"""

    return str(eval(expression))


def main():
    credentials = load_credentials()
    provider = CodexProvider(
        credentials=credentials,
        model="gpt-5.3-codex",
        on_credentials_refresh=save_credentials,
    )
    agent = Agent(provider=provider, tools=[get_weather, calculate])

    response = agent.run_sync("What's the weather in Tokyo?")
    print(f"Weather: {response.content}")

    response = agent.run_sync("What is 123 * 456?")
    print(f"Math: {response.content}")


if __name__ == "__main__":
    main()
