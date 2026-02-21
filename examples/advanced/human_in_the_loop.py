"""Human-in-the-loop examples with approval and feedback."""

import asyncio
import os

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider
from barebone import tool
from barebone.types import ToolCall
from barebone.types import ToolResult

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (requires approval)."""
    return f"Email sent to {to}"


@tool
def delete_file(path: str) -> str:
    """Delete a file (requires approval)."""
    return f"Deleted {path}"


@tool
def read_file(path: str) -> str:
    """Read a file (safe operation)."""
    return f"Contents of {path}"


REQUIRES_APPROVAL = {"send_email", "delete_file"}


def get_human_approval(tool_call: ToolCall) -> tuple[bool, str]:
    """Get human approval for a tool call."""
    print(f"\n{'=' * 40}")
    print(f"APPROVAL REQUIRED: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"{'=' * 40}")

    response = input("Approve? (y/n): ").strip().lower()

    if response == "y":
        return True, ""
    else:
        reason = input("Reason for rejection: ")
        return False, reason


async def human_approval_loop():
    """Agent loop with human approval for sensitive operations."""
    print("=" * 60)
    print("Human-in-the-Loop Approval")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    # Custom agent that checks for approval
    agent = Agent(
        provider=provider,
        tools=[send_email, delete_file, read_file],
        system="You are a helpful assistant. Use tools as needed.",
    )

    query = "Send an email to test@example.com with subject 'Hello' and body 'Hi there!'"
    print(f"\nQuery: {query}")

    # Manual loop with approval
    from barebone.types import Message

    agent.messages.append(Message(role="user", content=query))

    for _ in range(5):
        response = await provider.complete(
            messages=agent.messages,
            tools=agent.tools,
            system=agent.system,
        )

        if not response.tool_calls:
            print(f"\nFinal response: {response.content}")
            break

        agent.messages.append(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        results = []
        for tc in response.tool_calls:
            if tc.name in REQUIRES_APPROVAL:
                approved, reason = get_human_approval(tc)
                if not approved:
                    results.append(ToolResult(
                        id=tc.id,
                        content=f"REJECTED: {reason}",
                        is_error=True,
                    ))
                    continue

            # Execute the tool
            result = await agent._execute_tool(tc)
            print(f"Executed {tc.name}: {result.content}")
            results.append(result)

        agent.messages.append(Message(role="user", tool_results=results))


async def human_feedback_loop():
    """Iterative feedback loop for content improvement."""
    print("\n" + "=" * 60)
    print("Human Feedback Loop")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    task = "Write a product description for a smart water bottle"
    print(f"\nTask: {task}")

    response = await agent.run(task)
    output = response.content

    while True:
        print(f"\nCurrent output:\n{output}\n")
        feedback = input("Feedback (or 'done'): ").strip()

        if feedback.lower() == "done":
            break

        response = await agent.run(f"""Revise based on feedback.

Original task: {task}

Current output:
{output}

Feedback: {feedback}

Revised:""")
        output = response.content

    print(f"\nFinal output:\n{output}")
    return output


async def human_choice_branch():
    """Present options and let human choose the path."""
    print("\n" + "=" * 60)
    print("Human Choice Branch")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider)

    query = "Help me plan a weekend trip"
    print(f"\nQuery: {query}")

    # Get options
    response = await agent.run(f"""For this request, provide 3 different approaches.
Number them 1, 2, 3.

Request: {query}""")

    print(f"\nOptions:\n{response.content}\n")

    choice = input("Choose option (1/2/3): ").strip()

    # Execute chosen option
    response = await agent.run(f"""Execute option {choice} for: {query}

The options were:
{response.content}

Now implement option {choice} in detail:""")

    print(f"\nResult:\n{response.content}")
    return response.content


async def main():
    print("Example 1: Human Approval Loop")
    print("-" * 40)
    await human_approval_loop()

    print("\n\nExample 2: Human Feedback Loop")
    print("-" * 40)
    await human_feedback_loop()

    print("\n\nExample 3: Human Choice Branch")
    print("-" * 40)
    await human_choice_branch()


if __name__ == "__main__":
    asyncio.run(main())
