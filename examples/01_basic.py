import os

from barebone import Agent

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENAI_AUTH_KEY")


def main():
    agent = Agent(api_key=API_KEY, model="gpt-5.3-codex")
    response = agent.run_sync("What is the capital of France?")
    print(response.content)


if __name__ == "__main__":
    main()
