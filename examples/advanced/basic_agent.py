"""
Basic agent example using bare.

This example demonstrates:
1. Creating an agent with built-in tools
2. Running the agent with a prompt
3. Streaming responses
4. Using hooks for tool control
"""

import asyncio
import os

from dotenv import load_dotenv

from barebone import (
    Agent,
    HookContext,
    HookEvent,
    HookRegistry,
    HookResult,
    Router,
    tool,
)

# Load environment variables
load_dotenv()


# Create a custom tool
@tool("greet", "Greet a user by name")
def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}! Nice to meet you."


@tool("calculate", "Perform a calculation")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Only for demo, don't use eval in production!
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


async def basic_example():
    """Basic agent usage."""
    print("=" * 60)
    print("Basic Agent Example")
    print("=" * 60)

    # Create router with Anthropic OAuth
    router = Router(
        anthropic_oauth=os.getenv("ANTHROPIC_OAUTH_TOKEN"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openrouter=os.getenv("OPENROUTER_API_KEY"),
    )

    # Create agent with built-in tools
    agent = Agent(
        router=router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Be concise.",
        tools=["Read", "Glob"],  # Built-in tools by name
    )

    # Add custom tools
    agent.add_tool(greet_user)
    agent.add_tool(calculate)

    # Run the agent
    response = await agent.run("What files are in the current directory? Just list 3.")
    print(f"\nResponse: {response.content}")
    print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


async def streaming_example():
    """Streaming agent responses."""
    print("\n" + "=" * 60)
    print("Streaming Example")
    print("=" * 60)

    router = Router(
        anthropic_oauth=os.getenv("ANTHROPIC_OAUTH_TOKEN"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    agent = Agent(
        router=router,
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

    # Create hook registry
    hooks = HookRegistry()

    # Add a pre-tool hook that logs tool calls
    def log_tool_call(context: HookContext) -> HookResult:
        print(f"  [Hook] Tool called: {context.tool_name}")
        print(f"  [Hook] Input: {context.tool_input}")
        return HookResult(allow=True)

    hooks.register_pre_tool_use(".*", log_tool_call)  # Match all tools

    # Add a hook that denies certain operations
    def deny_dangerous_ops(context: HookContext) -> HookResult:
        if context.tool_name == "Bash":
            command = context.tool_input.get("command", "")
            if "rm " in command or "sudo" in command:
                return HookResult(allow=False, error="Dangerous command blocked")
        return HookResult(allow=True)

    hooks.register_pre_tool_use("Bash", deny_dangerous_ops)

    router = Router(
        anthropic_oauth=os.getenv("ANTHROPIC_OAUTH_TOKEN"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    agent = Agent(
        router=router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant.",
        tools=["Read", "Glob"],
        hooks=hooks,
    )

    response = await agent.run("List files in the current directory.")
    print(f"\nResponse: {response.content}")


async def subagent_example():
    """Using subagents for task delegation."""
    print("\n" + "=" * 60)
    print("Subagent Example")
    print("=" * 60)

    from barebone import SubagentDefinition, SubagentRegistry

    # Create subagent registry
    subagents = SubagentRegistry()

    # Define a code review subagent
    reviewer = SubagentDefinition(
        name="code-reviewer",
        description="Expert code reviewer for quality checks",
        system_prompt="""You are a code reviewer. Analyze code for:
1. Bugs and potential issues
2. Code style and readability
3. Performance concerns

Be concise and actionable.""",
        tools=["Read", "Glob", "Grep"],
        max_turns=5,
    )
    subagents.register(reviewer)

    router = Router(
        anthropic_oauth=os.getenv("ANTHROPIC_OAUTH_TOKEN"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    agent = Agent(
        router=router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant that can delegate tasks to specialists.",
        tools=["Read", "Glob"],
        subagents=subagents,
    )

    # The agent can now spawn subagents via the Task tool
    # (Task tool is automatically available when subagents are registered)
    print("Agent configured with code-reviewer subagent")
    print("Available subagents:", subagents.list())


async def conversation_example():
    """Multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Conversation Example")
    print("=" * 60)

    router = Router(
        anthropic_oauth=os.getenv("ANTHROPIC_OAUTH_TOKEN"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    agent = Agent(
        router=router,
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
    await subagent_example()
    await conversation_example()


if __name__ == "__main__":
    asyncio.run(main())
