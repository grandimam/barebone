import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import tool

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


agents = {}


@tool
def transfer_to(agent_name: str, context: str) -> str:
    return f"TRANSFER:{agent_name}:{context}"


def create_agents():
    agents["triage"] = Agent(
        api_key=API_KEY,
        model=MODEL,
        tools=[transfer_to],
        system="""You are a triage agent. Classify the customer issue and transfer to the right agent.

Available agents:
- support: General questions and how-to help
- billing: Payment, refunds, subscription issues
- engineering: Bugs, technical issues, errors

Use transfer_to(agent_name, context) to hand off.""",
    )

    agents["support"] = Agent(
        api_key=API_KEY,
        model=MODEL,
        tools=[transfer_to],
        system="""You are a support agent. Help with general questions.
If issue is billing-related, transfer to billing.
If issue is technical/bug, transfer to engineering.""",
    )

    agents["billing"] = Agent(
        api_key=API_KEY,
        model=MODEL,
        tools=[transfer_to],
        system="""You are a billing specialist. Handle payment and subscription issues.
If issue needs technical investigation, transfer to engineering.""",
    )

    agents["engineering"] = Agent(
        api_key=API_KEY,
        model=MODEL,
        system="You are a technical support engineer. Diagnose and solve technical issues.",
    )


async def handoff(query: str, max_transfers: int = 3):
    create_agents()

    current_agent = "triage"
    context = query

    for i in range(max_transfers + 1):
        print(f"\n[{current_agent.upper()}]")

        agent = agents[current_agent]
        response = await agent.run(context)

        if response.content and response.content.startswith("TRANSFER:"):
            parts = response.content.split(":", 2)
            next_agent = parts[1]
            transfer_context = parts[2] if len(parts) > 2 else context

            print(f"Transferring to {next_agent}...")
            current_agent = next_agent
            context = f"Transferred from previous agent. Context: {transfer_context}\n\nOriginal query: {query}"
        else:
            print(f"Response: {response.content}")
            return response.content

    return "Max transfers reached"


async def main():
    queries = [
        "I can't login to my account and I was charged twice",
        "How do I export my data?",
        "The app crashes when I click the submit button",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Customer: {query}")
        print("=" * 60)
        await handoff(query)


if __name__ == "__main__":
    asyncio.run(main())
