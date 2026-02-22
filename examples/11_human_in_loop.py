import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import Question
from barebone import QuestionOption
from barebone import ask_user_question
from barebone import tool

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@tool
def execute_action(action: str) -> str:
    return f"Executed: {action}"


async def main():
    agent = Agent(
        api_key=API_KEY,
        model="claude-sonnet-4-20250514",
        tools=[ask_user_question, execute_action],
        system="Before taking important actions, ask the user for confirmation.",
    )

    response = await agent.run("Delete the old backup files")
    print(f"Result: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
