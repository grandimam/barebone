import asyncio

from barebone import (
    agent,
    AgentEvent,
    AgentRunner,
    Context,
    FileStorage,
)


@agent
async def multi_step(ctx: Context):
    """Agent that processes steps and can be resumed."""
    ctx.state.setdefault("step", 0)
    ctx.state.setdefault("results", [])

    steps = ["fetch", "process", "validate", "complete"]

    while ctx.state["step"] < len(steps):
        current = steps[ctx.state["step"]]
        print(f"Step {ctx.state['step'] + 1}/{len(steps)}: {current}")

        await asyncio.sleep(1)
        ctx.state["results"].append(f"{current}_done")
        ctx.state["step"] += 1

        await ctx.checkpoint()

        if current == "process":
            print("  [Waiting for validation approval...]")
            response = await ctx.suspend(reason="Approve validation")
            print(f"  [Got: {response}]")

    return {"steps": ctx.state["step"], "results": ctx.state["results"]}


async def run_fresh():
    """Start a new agent."""
    runner = AgentRunner(storage=FileStorage("./resume_data"))

    print("=== Starting fresh agent ===")
    handle = await runner.start(
        multi_step,
        agent_id="multi-001",
    )

    await asyncio.sleep(3)

    if handle.status.value == "suspended":
        print("\n[Agent is suspended, stopping without approval...]")
        await runner.stop("multi-001")
        print("[Agent stopped. State saved to disk.]")
        return False

    result = await handle.wait()
    print(f"Result: {result}")
    return True


async def run_resume():
    """Resume an existing agent."""
    runner = AgentRunner(storage=FileStorage("./resume_data"))

    checkpoints = await runner.list_checkpoints()
    print(f"=== Found checkpoints: {checkpoints} ===")

    if "multi-001" not in checkpoints:
        print("No checkpoint found. Run fresh first.")
        return

    print("=== Resuming agent ===")
    handle = await runner.resume(multi_step, agent_id="multi-001")

    if not handle:
        print("Failed to resume")
        return

    print(f"Resumed at step: {handle.state.get('step')}")

    await asyncio.sleep(1)
    if handle.status.value == "suspended":
        print("[Sending approval...]")
        handle.send(AgentEvent(type="approve", data={"ok": True}))

    result = await handle.wait()
    print(f"Result: {result}")


async def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        await run_resume()
    else:
        completed = await run_fresh()
        if not completed:
            print("\nRun with 'resume' argument to continue:")
            print("  python resume_agent.py resume")


if __name__ == "__main__":
    asyncio.run(main())
