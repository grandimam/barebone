"""Advanced example demonstrating hooks, subagents, and skills."""

import asyncio
import os

from barebone import (
    Agent,
    Router,
    TokenManager,
    HookRegistry,
    HookEvent,
    HookContext,
    HookResult,
    SubagentRegistry,
    SubagentDefinition,
    BUILTIN_SUBAGENTS,
    tool,
    Done,
    TextDelta,
)


@tool
def search_code(query: str) -> str:
    """Search for code patterns in the codebase."""
    return f"Found 3 matches for '{query}' in src/utils.py, src/main.py, src/config.py"


# Hook handler - runs before any tool execution
def log_tool_use(context: HookContext) -> HookResult:
    """Log all tool usage."""
    print(f"  [HOOK] Tool: {context.tool_name}, Input: {list(context.tool_input.keys())}")
    return HookResult(allow=True)


# Hook handler - block dangerous operations
def security_check(context: HookContext) -> HookResult:
    """Block dangerous file operations."""
    if context.tool_name in ("Write", "Edit"):
        file_path = context.tool_input.get("file_path", "")
        if "/etc/" in file_path or "/root/" in file_path:
            return HookResult(
                allow=False,
                error=f"Blocked: Cannot modify system file {file_path}",
            )
    return HookResult(allow=True)


async def main():
    # Setup with automatic OAuth
    token_manager = TokenManager.auto()

    if not token_manager.has_credentials:
        print("No OAuth credentials found. Please run Claude Code and login first.")
        return

    oauth_token = await token_manager.get_token()
    router = Router(anthropic_oauth=oauth_token)

    # Setup hooks
    hooks = HookRegistry()
    hooks.register_pre_tool_use(".*", log_tool_use)  # Log all tools
    hooks.register_pre_tool_use("Write|Edit", security_check)  # Security check for file ops

    # Setup subagents
    subagents = SubagentRegistry()

    # Register built-in subagents
    for name, definition in BUILTIN_SUBAGENTS.items():
        subagents.register(definition)

    # Register a custom subagent
    code_reviewer = SubagentDefinition(
        name="code-reviewer",
        description="Reviews code for quality, security, and best practices.",
        system_prompt="""You are an expert code reviewer. Analyze code for:
1. Security vulnerabilities
2. Performance issues
3. Code style and best practices
4. Potential bugs

Provide concise, actionable feedback.""",
        tools=["Read", "Glob", "Grep"],
        model="claude-sonnet-4-20250514",
        max_turns=5,
    )
    subagents.register(code_reviewer)

    # Create the main agent
    agent = Agent(
        router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful coding assistant with access to tools and subagents.",
        tools=["Read", "Glob", "Grep", search_code],
        hooks=hooks,
        subagents=subagents,
    )

    # Subscribe to agent events
    def on_event(event):
        if event.type == "tool_start":
            print(f"  [EVENT] Starting tool: {event.data.get('tool_name')}")
        elif event.type == "subagent_start":
            print(f"  [EVENT] Spawning subagent: {event.data.get('agent_type')}")

    agent.subscribe(on_event)

    # Example 1: Simple tool use with hooks
    print("\n=== Example 1: Tool Use with Hooks ===")
    result = await agent.run("Search for 'async def' in the codebase")
    print(f"Response: {result.content[:200]}...")

    # Example 2: Streaming with events
    print("\n=== Example 2: Streaming ===")
    async for event in agent.stream("List all Python files"):
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case Done():
                print()

    # Cleanup
    await token_manager.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
