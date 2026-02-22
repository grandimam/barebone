import asyncio
import os

from barebone import Agent
from barebone import CodexProvider
from barebone import OAuthCredentials

from dotenv import load_dotenv

load_dotenv()


OPENAI_ACCESS_TOKEN = os.environ.get('OPENAI_ACCESS_TOKEN')
OPENAI_REFRESH_TOKEN = os.environ.get('OPENAI_REFRESH_TOKEN')
OPENAI_ACCOUNT_ID = os.environ.get('OPENAI_ACCOUNT_ID')

async def main():

    credentials = OAuthCredentials(
        access_token=OPENAI_ACCESS_TOKEN,
        refresh_token=OPENAI_REFRESH_TOKEN,
        expires_at=0,
        account_id=OPENAI_ACCOUNT_ID,
    )

    provider = CodexProvider(credentials=credentials, model="gpt-5.3-codex")
    agent = Agent(provider=provider)

    response = await agent.run("What is 2 + 2?")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
