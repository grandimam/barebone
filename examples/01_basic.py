import os

from dotenv import load_dotenv

from barebone import Agent

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def main():
    agent = Agent(api_key=API_KEY, model="claude-sonnet-4-20250514")
    response = agent.run_sync("What is the capital of France?")
    print(response.content)


if __name__ == "__main__":
    main()
