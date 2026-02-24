import asyncio
import os

from barebone import (
    agent,
    AgentEvent,
    AgentRunner,
    Context,
    FileStorage,
    LLMClient,
    Message,
    Request,
    TextDelta,
    Done,
    tool,
)


@tool
def search(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"


@agent
async def researcher(ctx: Context):
    ctx.state.setdefault("phase", "gather")
    ctx.state.setdefault("sources", [])
    ctx.state.setdefault("messages", [])

    llm = LLMClient(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model="anthropic/claude-sonnet-4",
    )

    topic = ctx.state.get("topic", "AI agents")

    if ctx.state["phase"] == "gather":
        ctx.state["messages"].append(
            Message(role="user", content=f"Find 3 key facts about: {topic}")
        )

        request = Request(
            id="gather",
            messages=ctx.state["messages"],
            system="You are a research assistant. Be concise.",
        )

        content = ""
        async for event in llm.stream(request):
            if isinstance(event, TextDelta):
                content += event.text
                print(event.text, end="", flush=True)
            elif isinstance(event, Done):
                ctx.state["sources"].append(content)
                ctx.state["messages"].append(
                    Message(role="assistant", content=content)
                )

        print("\n")
        ctx.state["phase"] = "review"
        await ctx.checkpoint()

    if ctx.state["phase"] == "review":
        print(f"[Agent suspended - waiting for approval]")
        print(f"Sources found: {len(ctx.state['sources'])}")

        approved = await ctx.suspend(reason="Review sources", timeout=60.0)

        if approved:
            ctx.state["approved"] = True
            print(f"[Approved with data: {approved}]")
        else:
            ctx.state["approved"] = False
            print("[Timeout - auto-approved]")

        ctx.state["phase"] = "summarize"
        await ctx.checkpoint()

    if ctx.state["phase"] == "summarize":
        ctx.state["messages"].append(
            Message(role="user", content="Summarize the key points in one sentence.")
        )

        request = Request(
            id="summarize",
            messages=ctx.state["messages"],
            system="You are a research assistant. Be concise.",
        )

        summary = ""
        async for event in llm.stream(request):
            if isinstance(event, TextDelta):
                summary += event.text
                print(event.text, end="", flush=True)
            elif isinstance(event, Done):
                ctx.state["summary"] = summary

        print("\n")
        ctx.state["phase"] = "done"

    await llm.close()
    return {
        "topic": topic,
        "summary": ctx.state.get("summary"),
        "approved": ctx.state.get("approved"),
    }


async def main():
    runner = AgentRunner(storage=FileStorage("./agent_data"))

    print("Starting researcher agent...")
    print("-" * 50)

    handle = await runner.start(
        researcher,
        state={"topic": "Python asyncio"},
        agent_id="research-001",
    )

    await asyncio.sleep(2)

    if handle.status.value == "suspended":
        print("\n[Sending approval event...]")
        handle.send(AgentEvent(type="approved", data={"reviewer": "human"}))

    result = await handle.wait()

    print("-" * 50)
    print(f"Final result: {result}")
    print(f"Agent status: {handle.status}")


if __name__ == "__main__":
    asyncio.run(main())
