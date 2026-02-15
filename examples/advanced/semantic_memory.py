"""Example demonstrating semantic memory search."""

import asyncio
import os

from barebone import (
    Agent,
    Router,
    TokenManager,
    SemanticMemory,
    MemoryConfig,
)


# Sample documents to index
SAMPLE_DOCS = {
    "auth.md": """# Authentication Guide

## OAuth Flow
bare supports automatic OAuth token discovery from Claude Code.
The TokenManager.auto() method checks:
1. macOS Keychain
2. ~/.claude/.credentials.json
3. ~/.bare/credentials.json

## API Keys
You can also use traditional API keys:
- ANTHROPIC_API_KEY for Claude
- OPENROUTER_API_KEY for other models

## Token Refresh
Tokens are automatically refreshed when expired. The refresh happens
transparently when you call get_token().
""",
    "tools.md": """# Tools Guide

## Built-in Tools
bare provides these built-in tools:
- Read: Read file contents
- Write: Write to files
- Edit: Edit files with string replacement
- Bash: Execute shell commands
- Glob: Find files by pattern
- Grep: Search file contents

## Custom Tools
Use the @tool decorator to create custom tools:

```python
@tool
def my_tool(param: str) -> str:
    return f"Result: {param}"
```

## Tool Hooks
Use hooks to intercept tool calls:
- PreToolUse: Before tool execution
- PostToolUse: After tool execution
""",
    "agents.md": """# Agent Guide

## Creating Agents
Create an agent with Router and model:

```python
agent = Agent(
    router,
    model="claude-sonnet-4",
    system="You are helpful.",
    tools=["Read", "Glob"],
)
```

## Subagents
Spawn specialized subagents for complex tasks:
- Explore: Fast code exploration
- Bash: Command execution
- general-purpose: Complex multi-step tasks

## Memory
Agents can use semantic memory for knowledge retrieval.
The memory_search tool finds relevant information using
hybrid vector + keyword search.
""",
}


async def main():
    # Setup
    token_manager = TokenManager.auto()
    if not token_manager.has_credentials:
        print("No OAuth credentials found")
        return

    # Check for API keys (needed for embeddings)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openrouter_key and not openai_key:
        print("OPENROUTER_API_KEY or OPENAI_API_KEY required for embeddings")
        print("Set with: export OPENROUTER_API_KEY=sk-or-...")
        return

    token = await token_manager.get_token()
    router = Router(anthropic_oauth=token, openrouter=openrouter_key)

    # Create semantic memory (prefer OpenRouter if available)
    print("=== Setting up Semantic Memory ===")
    embedding_provider = "openrouter" if openrouter_key else "openai"
    print(f"Using {embedding_provider} for embeddings")

    memory = SemanticMemory(
        config=MemoryConfig(
            embedding_provider=embedding_provider,
            embedding_api_key=openrouter_key or openai_key,
            chunk_size=200,  # Smaller chunks for demo
            chunk_overlap=40,
        )
    )

    # Index sample documents
    print("\nIndexing documents...")
    for name, content in SAMPLE_DOCS.items():
        chunks = await memory.index_text(content, source=name)
        print(f"  {name}: {chunks} chunks")

    # Show stats
    stats = memory.stats()
    print(f"\nMemory stats: {stats['chunks']} chunks indexed")

    # Test search
    print("\n=== Testing Search ===")

    queries = [
        "How do I authenticate?",
        "What tools are available?",
        "How do I create custom tools?",
        "How do subagents work?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = await memory.search(query, limit=2)
        for result in results:
            print(f"  [{result.score:.2f}] {result.source}: {result.text[:80]}...")

    # Use with agent
    print("\n=== Using Memory with Agent ===")

    agent = Agent(
        router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant with access to documentation memory.",
        tools=memory.get_tools(),  # Add memory_search and memory_get tools
    )

    # Ask a question that requires memory search
    response = await agent.run(
        "Search the memory and tell me: How does token refresh work in bare?"
    )
    print(f"\nAgent response:\n{response.content}")

    # Cleanup
    await memory.close()
    await token_manager.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
