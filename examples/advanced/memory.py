"""Example demonstrating conversation memory and persistence."""

import asyncio

from barebone import Agent, Router, TokenManager, Memory, SQLiteBackend


async def main():
    # Setup
    token_manager = TokenManager.auto()
    if not token_manager.has_credentials:
        print("No OAuth credentials found")
        return

    token = await token_manager.get_token()
    router = Router(anthropic_oauth=token)

    # Create memory with SQLite backend (persistent)
    # Default FileBackend also works: memory = Memory()
    memory = Memory(backend=SQLiteBackend())

    # Check for existing conversations
    conversations = memory.list_conversations(limit=5)
    if conversations:
        print("Recent conversations:")
        for i, conv in enumerate(conversations):
            title = conv.title or "(untitled)"
            print(f"  {i + 1}. {title} - {conv.id[:8]}...")

    # Create or resume conversation
    print("\n=== Starting New Conversation ===")
    conv_id = memory.new_conversation(title="Weather Chat")
    print(f"Conversation ID: {conv_id}")

    # Create agent
    agent = Agent(
        router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Remember our conversation context.",
    )

    # First message
    print("\n[Turn 1]")
    response = await agent.run("Hi! My name is Alex and I live in Seattle.")
    print(f"Assistant: {response.content}")

    # Save after each turn
    memory.save(conv_id, agent.messages)

    # Second message - agent should remember context
    print("\n[Turn 2]")
    response = await agent.run("What's typically the weather like where I live?")
    print(f"Assistant: {response.content}")

    # Save again
    memory.save(conv_id, agent.messages)

    # Show how to resume later
    print("\n=== Simulating Resume Later ===")

    # Create new agent (simulating new session)
    new_agent = Agent(
        router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Remember our conversation context.",
    )

    # Load previous messages
    previous_messages = memory.load(conv_id)
    print(f"Loaded {len(previous_messages)} previous messages")

    # Continue conversation with context
    print("\n[Turn 3 - Resumed]")
    response = await new_agent.run(
        "What was my name again?",
        messages=previous_messages,
    )
    print(f"Assistant: {response.content}")

    # Save the continued conversation
    memory.save(conv_id, new_agent.messages)

    # List final state
    print("\n=== Conversation History ===")
    final_messages = memory.load(conv_id)
    for msg in final_messages:
        role = msg.role.upper()
        content = msg.content if isinstance(msg.content, str) else "(tool use)"
        print(f"  [{role}] {content[:80]}...")

    # Cleanup
    await token_manager.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
