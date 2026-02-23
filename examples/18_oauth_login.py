import asyncio

from barebone import Agent
from barebone import CodexProvider
from barebone import load_credentials
from barebone import save_credentials


async def main():
    credentials = load_credentials()

    provider = CodexProvider(
        credentials=credentials,
        model="gpt-5.3-codex",
        on_credentials_refresh=save_credentials,
    )
    agent = Agent(provider=provider)

    response = await agent.run("What is 2 + 2?")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
