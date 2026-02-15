import asyncio

from dotenv import load_dotenv

from barebone import Agent, Hooks, tool

load_dotenv()

@tool
def greet_user(name: str) -> str:
    return f"Hello, {name}! Nice to meet you."


@tool
def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


async def basic_example():
    print("=" * 60)
    print("Basic Agent Example")
    print("=" * 60)

    agent = Agent(
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Be concise.",
        tools=["Read", "Glob", greet_user, calculate],
    )

    response = await agent.run("What files are in the current directory? Just list 3.")
    print(f"\nResponse: {response.content}")
    print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


async def streaming_example():
    """Streaming agent responses."""
    print("\n" + "=" * 60)
    print("Streaming Example")
    print("=" * 60)

    agent = Agent(
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant.",
    )

    print("\nStreaming response:")
    async for event in agent.stream("Write a haiku about coding."):
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)
    print()


async def hooks_example():
    """Using hooks to control tool execution."""
    print("\n" + "=" * 60)
    print("Hooks Example")
    print("=" * 60)

    # Create hooks
    hooks = Hooks()

    @hooks.before
    def log_tool_call(tool_call):
        print(f"  [Hook] Tool called: {tool_call.name}")
        print(f"  [Hook] Args: {tool_call.arguments}")

    @hooks.before
    def deny_dangerous_ops(tool_call):
        if tool_call.name == "Bash":
            command = tool_call.arguments.get("command", "")
            if "rm " in command or "sudo" in command:
                raise Hooks.Deny("Dangerous command blocked")

    @hooks.after
    def log_result(tool_call, result):
        print(f"  [Hook] Result: {result[:100]}...")

    agent = Agent(
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant.",
        tools=["Read", "Glob"],
        hooks=hooks,
    )

    response = await agent.run("List files in the current directory.")
    print(f"\nResponse: {response.content}")


async def conversation_example():
    """Multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Conversation Example")
    print("=" * 60)

    agent = Agent(
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Remember our conversation.",
    )

    # First turn
    response = await agent.run("My name is Alice.")
    print(f"Turn 1: {response.content}")

    # Second turn (agent remembers the conversation)
    response = await agent.run("What's my name?")
    print(f"Turn 2: {response.content}")

    # Clear conversation
    agent.clear_messages()
    response = await agent.run("What's my name?")
    print(f"Turn 3 (after clear): {response.content}")


async def main():
    """Run all examples."""
    await basic_example()
    await streaming_example()
    await hooks_example()
    await conversation_example()


if __name__ == "__main__":
    asyncio.run(main())
