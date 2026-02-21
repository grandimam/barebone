"""Multi-agent collaboration examples."""

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from barebone import Agent
from barebone import AnthropicProvider

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"


@dataclass
class AgentConfig:
    name: str
    role: str
    system: str


AGENTS = {
    "researcher": AgentConfig(
        name="Researcher",
        role="research",
        system="You are a thorough researcher. Find facts and cite sources. Be objective.",
    ),
    "writer": AgentConfig(
        name="Writer",
        role="writing",
        system="You are a creative writer. Write engaging, clear content. Focus on readability.",
    ),
    "critic": AgentConfig(
        name="Critic",
        role="review",
        system="You are a constructive critic. Find issues and suggest improvements. Be specific.",
    ),
    "editor": AgentConfig(
        name="Editor",
        role="editing",
        system="You are a meticulous editor. Fix grammar, improve clarity, ensure consistency.",
    ),
}


async def agent_call(agent_name: str, message: str) -> str:
    """Call a specific agent with a message."""
    config = AGENTS[agent_name]
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    agent = Agent(provider=provider, system=config.system)
    response = await agent.run(message)
    return response.content


async def pipeline_collaboration(topic: str) -> str:
    """Sequential pipeline: researcher -> writer -> editor."""
    print("=" * 60)
    print("Pipeline Collaboration")
    print("=" * 60)

    print("\n[Researcher]")
    research = await agent_call("researcher", f"Research key facts about: {topic}")
    print(f"Research: {research[:200]}...")

    print("\n[Writer]")
    draft = await agent_call("writer", f"Write a short article using this research:\n\n{research}")
    print(f"Draft: {draft[:200]}...")

    print("\n[Editor]")
    final = await agent_call("editor", f"Edit and polish this article:\n\n{draft}")
    print(f"Final: {final[:200]}...")

    return final


async def debate_collaboration(topic: str) -> str:
    """Debate pattern: pro vs con -> synthesis."""
    print("\n" + "=" * 60)
    print("Debate Collaboration")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    print("\n[Pro Agent]")
    pro_agent = Agent(
        provider=provider,
        system="You argue FOR positions. Be persuasive but factual.",
    )
    pro = (await pro_agent.run(f"Argue in favor of: {topic}")).content
    print(f"Pro: {pro[:150]}...")

    print("\n[Con Agent]")
    con_agent = Agent(
        provider=provider,
        system="You argue AGAINST positions. Be persuasive but factual.",
    )
    con = (await con_agent.run(f"Argue against: {topic}")).content
    print(f"Con: {con[:150]}...")

    print("\n[Synthesizer]")
    synth_agent = Agent(
        provider=provider,
        system="You synthesize opposing views into balanced, nuanced conclusions.",
    )
    synthesis = (await synth_agent.run(
        f"Synthesize these opposing views into a balanced conclusion:\n\nPro: {pro}\n\nCon: {con}"
    )).content
    print(f"Synthesis: {synthesis[:150]}...")

    return synthesis


async def parallel_specialists(question: str) -> str:
    """Parallel specialists providing different perspectives."""
    print("\n" + "=" * 60)
    print("Parallel Specialists")
    print("=" * 60)

    specialists = {
        "technical": "You are a technical expert. Focus on how things work.",
        "practical": "You are practical-minded. Focus on real-world applications.",
        "historical": "You are a historian. Focus on origins and evolution.",
    }

    async def get_perspective(name: str, system: str) -> tuple[str, str]:
        provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
        agent = Agent(provider=provider, system=system)
        response = await agent.run(f"Answer from your perspective: {question}")
        return name, response.content

    # Run all specialists in parallel
    tasks = [get_perspective(name, system) for name, system in specialists.items()]
    results = await asyncio.gather(*tasks)

    print("\nPerspectives gathered:")
    for name, content in results:
        print(f"  [{name}]: {content[:80]}...")

    # Combine perspectives
    combined = "\n\n".join(f"[{name}]: {content}" for name, content in results)
    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)
    final_agent = Agent(provider=provider)
    final = await final_agent.run(
        f"Combine these expert perspectives into a comprehensive answer:\n\n{combined}"
    )

    print(f"\nCombined: {final.content[:200]}...")
    return final.content


async def hierarchical_agents(task: str) -> str:
    """Hierarchical pattern: manager delegates to workers."""
    print("\n" + "=" * 60)
    print("Hierarchical Agents")
    print("=" * 60)

    provider = AnthropicProvider(api_key=API_KEY, model=MODEL)

    print("\n[Manager]")
    manager = Agent(
        provider=provider,
        system="You are a manager. Delegate effectively and review work.",
    )
    breakdown = (await manager.run(
        f"Break this into 2-3 subtasks for your team:\n\n{task}"
    )).content
    print(f"Breakdown:\n{breakdown}")

    print("\n[Worker]")
    worker = Agent(
        provider=provider,
        system="You are a diligent worker. Complete tasks thoroughly.",
    )
    work = (await worker.run(f"Complete these assigned tasks:\n\n{breakdown}")).content
    print(f"Work completed: {work[:200]}...")

    print("\n[Manager Review]")
    review = (await manager.run(
        f"Review this work and provide final output:\n\nOriginal task: {task}\n\nCompleted work:\n{work}"
    )).content
    print(f"Final review: {review[:200]}...")

    return review


async def main():
    await pipeline_collaboration("the impact of AI on employment")
    await debate_collaboration("Remote work is better than office work")
    await parallel_specialists("What is machine learning?")
    await hierarchical_agents("Create a marketing plan for a new mobile app")


if __name__ == "__main__":
    asyncio.run(main())
