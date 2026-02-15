"""Comprehensive example demonstrating all bare features."""

import asyncio
import os

from barebone import (
    # Core
    Agent,
    Router,
    TokenManager,
    tool,
    # Tracking
    UsageTracker,
    TokenUsage,
    # Transcripts
    Transcript,
    # Compaction
    CompactionConfig,
    ContextCompactor,
    CompactionState,
    # Approval
    ApprovalManager,
    ApprovalLevel,
    ApprovalRule,
    # Thinking
    ThinkingLevel,
    build_thinking_params,
    # Session
    SessionManager,
    ResetPolicy,
    ModelFallback,
    ModelFallbackConfig,
    # Hooks
    HookRegistry,
)


@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a symbol."""
    # Mock data
    prices = {"AAPL": 185.50, "GOOGL": 141.25, "MSFT": 378.90}
    if symbol.upper() in prices:
        return f"{symbol.upper()}: ${prices[symbol.upper()]}"
    return f"Unknown symbol: {symbol}"


async def demo_tracking():
    """Demonstrate token and cost tracking."""
    print("\n" + "=" * 60)
    print("1. TOKEN & COST TRACKING")
    print("=" * 60)

    tracker = UsageTracker()
    session = tracker.start_session("demo-session")

    # Simulate some usage
    session.record_usage("claude-sonnet-4-20250514", TokenUsage(
        input_tokens=5000,
        output_tokens=2000,
    ))
    session.record_tool_call()
    session.record_tool_call()
    session.record_turn()

    print(session.summary())
    print(f"\nDetailed stats: {session.to_dict()}")


async def demo_transcript():
    """Demonstrate JSONL transcripts."""
    print("\n" + "=" * 60)
    print("2. JSONL TRANSCRIPTS")
    print("=" * 60)

    transcript = Transcript.create(
        session_id="demo-transcript",
        model="claude-sonnet-4",
    )

    # Log some messages
    transcript.log_user("What's the weather in Tokyo?")
    transcript.log_assistant(
        "The weather in Tokyo is sunny and 72°F.",
        usage=TokenUsage(input_tokens=50, output_tokens=20),
    )
    transcript.log_system("session_started", {"agent": "main"})

    # Show stats
    stats = transcript.get_stats()
    print(f"Transcript path: {stats['path']}")
    print(f"Messages: {stats['message_counts']}")
    print(f"Total tokens: {stats['total_tokens']}")


async def demo_compaction():
    """Demonstrate context compaction."""
    print("\n" + "=" * 60)
    print("3. CONTEXT COMPACTION")
    print("=" * 60)

    from barebone import Message

    config = CompactionConfig(
        max_context_tokens=10000,
        compaction_threshold=5000,
        strategy="truncate",
        keep_recent_turns=2,
    )

    compactor = ContextCompactor(config)
    state = CompactionState(total_tokens=6000)  # Over threshold

    # Create sample messages
    messages = [
        Message(role="user", content="Hello!"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="Tell me about Python."),
        Message(role="assistant", content="Python is a great language..."),
        Message(role="user", content="What about JavaScript?"),
        Message(role="assistant", content="JavaScript is widely used..."),
    ]

    print(f"Before compaction: {len(messages)} messages")
    print(f"Token estimate: {state.total_tokens}")

    compacted = await compactor.maybe_compact(messages, state)

    print(f"After compaction: {len(compacted)} messages")
    print(f"Compaction count: {state.compaction_count}")


async def demo_approval():
    """Demonstrate tool approval workflow."""
    print("\n" + "=" * 60)
    print("4. TOOL APPROVAL")
    print("=" * 60)

    # Custom approval handler (non-interactive for demo)
    async def auto_approve(request):
        from barebone import ApprovalResponse
        print(f"  [Auto-approving] {request.tool_name}: {request.reason}")
        return ApprovalResponse(approved=True)

    approval = ApprovalManager(
        rules=[
            ApprovalRule(
                tool_pattern="Bash",
                level=ApprovalLevel.CONFIRM,
                reason="Shell command execution",
            ),
        ],
        handler=auto_approve,
        use_default_rules=False,
    )

    # Check approval
    level, rule = approval.check_approval_needed("Bash", {"command": "ls -la"})
    print(f"Tool: Bash")
    print(f"Approval level: {level.value}")
    print(f"Reason: {rule.reason if rule else 'N/A'}")

    # Request approval
    response = await approval.request_approval("Bash", {"command": "ls"}, "tool-123")
    print(f"Approved: {response.approved}")


async def demo_thinking():
    """Demonstrate extended thinking configuration."""
    print("\n" + "=" * 60)
    print("5. EXTENDED THINKING")
    print("=" * 60)

    # Build thinking params for API
    params = build_thinking_params(ThinkingLevel.MEDIUM)
    print(f"Thinking level: MEDIUM")
    print(f"API params: {params}")

    params_high = build_thinking_params(ThinkingLevel.HIGH)
    print(f"\nThinking level: HIGH")
    print(f"API params: {params_high}")


async def demo_session():
    """Demonstrate session management."""
    print("\n" + "=" * 60)
    print("6. SESSION MANAGEMENT")
    print("=" * 60)

    policy = ResetPolicy(
        idle_timeout_seconds=3600,
        max_turns=100,
        max_tokens=500000,
    )

    manager = SessionManager(policy=policy)

    # Create session
    session = manager.get_session("user-123")
    print(f"Session ID: {session.session_id[:8]}...")
    print(f"Created at: {session.created_at}")

    # Simulate activity
    from barebone import Message
    session.add_message(Message(role="user", content="Hello"))
    session.add_message(Message(role="assistant", content="Hi!"))

    print(f"Turn count: {session.turn_count}")
    print(f"Message count: {len(session.messages)}")

    # Check reset policy
    should_reset, reason = manager.should_reset(session)
    print(f"Should reset: {should_reset} (reason: {reason})")


async def demo_fallback():
    """Demonstrate multi-model fallback."""
    print("\n" + "=" * 60)
    print("7. MULTI-MODEL FALLBACK")
    print("=" * 60)

    config = ModelFallbackConfig(
        primary_model="claude-sonnet-4-20250514",
        fallback_models=["claude-haiku-3-5-20241022", "gpt-4o-mini"],
        max_retries=2,
    )

    fallback = ModelFallback(config)

    print(f"Primary model: {config.primary_model}")
    print(f"Fallback chain: {config.fallback_models}")
    print(f"Max retries per model: {config.max_retries}")


async def demo_live_agent():
    """Live demonstration with actual API calls."""
    print("\n" + "=" * 60)
    print("8. LIVE AGENT DEMO")
    print("=" * 60)

    # Check for credentials
    token_manager = TokenManager.auto()
    if not token_manager.has_credentials:
        print("No OAuth credentials found. Skipping live demo.")
        return

    token = await token_manager.get_token()
    router = Router(anthropic_oauth=token)

    # Setup tracking
    tracker = UsageTracker()
    session_stats = tracker.start_session("live-demo")

    # Setup transcript
    transcript = Transcript.create(session_id="live-demo", model="claude-sonnet-4")

    # Setup hooks with approval
    hooks = HookRegistry()

    # Create agent
    agent = Agent(
        router,
        model="claude-sonnet-4-20250514",
        system="You are a helpful assistant. Be concise.",
        tools=[get_stock_price],
        hooks=hooks,
    )

    # Run a query
    print("\nRunning: 'What is Apple stock price?'")
    transcript.log_user("What is Apple stock price?")

    response = await agent.run("What is Apple stock price?")

    # Track usage
    session_stats.record_usage(
        "claude-sonnet-4-20250514",
        TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
    )

    transcript.log_assistant(response.content)

    print(f"\nResponse: {response.content}")
    print(f"\n{session_stats.summary()}")

    # Cleanup
    await token_manager.close()


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("LOOPFLOW - FULL FEATURES DEMONSTRATION")
    print("=" * 60)

    await demo_tracking()
    await demo_transcript()
    await demo_compaction()
    await demo_approval()
    await demo_thinking()
    await demo_session()
    await demo_fallback()
    await demo_live_agent()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
