from barebone import Agent
from barebone import CodexProvider
from barebone import load_credentials
from barebone import save_credentials


def main():
    credentials = load_credentials()
    provider = CodexProvider(
        credentials=credentials,
        model="gpt-5.3-codex",
        on_credentials_refresh=save_credentials,
    )
    agent = Agent(provider=provider)
    response = agent.run_sync("What is the capital of France?")
    print(response.content)


if __name__ == "__main__":
    main()
