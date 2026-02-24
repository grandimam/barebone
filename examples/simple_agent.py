import asyncio

from barebone import (
    agent,
    AgentEvent,
    AgentRunner,
    Context,
    MemoryStorage,
)


@agent
async def counter(ctx: Context):
    """Simple agent that counts and waits for approval at each step."""
    ctx.state.setdefault("count", 0)
    ctx.state.setdefault("max", 5)

    while ctx.state["count"] < ctx.state["max"]:
        ctx.state["count"] += 1
        print(f"Count: {ctx.state['count']}")

        await ctx.checkpoint()

        if ctx.state["count"] % 2 == 0:
            print(f"  [Waiting for approval at {ctx.state['count']}...]")
            response = await ctx.suspend(reason="Approve to continue", timeout=5.0)

            if response:
                print(f"  [Approved: {response}]")
            else:
                print(f"  [Timeout - continuing anyway]")

        await asyncio.sleep(0.5)

    return {"final_count": ctx.state["count"]}


async def main():
    runner = AgentRunner(storage=MemoryStorage())

    print("Starting counter agent...")
    print("-" * 30)

    handle = await runner.start(
        counter,
        state={"max": 6},
        agent_id="counter-001",
    )

    async def send_approvals():
        for i in range(3):
            await asyncio.sleep(2)
            if handle.status.value == "suspended":
                print(f"  [Sending approval #{i + 1}]")
                handle.send(AgentEvent(type="approve", data={"attempt": i + 1}))

    approval_task = asyncio.create_task(send_approvals())

    result = await handle.wait()
    approval_task.cancel()

    print("-" * 30)
    print(f"Result: {result}")
    print(f"Status: {handle.status}")


if __name__ == "__main__":
    asyncio.run(main())
