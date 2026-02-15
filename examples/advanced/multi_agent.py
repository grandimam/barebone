"""
Multi-Agent pattern.

Multiple agents with different roles collaborate on tasks.
"""

import asyncio
from dataclasses import dataclass

from barebone import acomplete, complete, user


@dataclass
class AgentConfig:
    name: str
    role: str
    system: str


# Define agent personas
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


def agent_call(agent_name: str, message: str) -> str:
    """Make a call as a specific agent."""
    agent = AGENTS[agent_name]
    response = complete(
        "claude-sonnet-4-20250514",
        [user(message)],
        system=agent.system,
    )
    return response.content


def pipeline_collaboration(topic: str) -> str:
    """Agents work in a pipeline: researcher -> writer -> editor."""
    print("=" * 60)
    print("Pipeline Collaboration")
    print("=" * 60)

    # Researcher gathers info
    print("\n[Researcher]")
    research = agent_call("researcher", f"Research key facts about: {topic}")
    print(f"Research: {research[:200]}...")

    # Writer creates content
    print("\n[Writer]")
    draft = agent_call("writer", f"Write a short article using this research:\n\n{research}")
    print(f"Draft: {draft[:200]}...")

    # Editor polishes
    print("\n[Editor]")
    final = agent_call("editor", f"Edit and polish this article:\n\n{draft}")
    print(f"Final: {final[:200]}...")

    return final


def debate_collaboration(topic: str) -> str:
    """Two agents debate, third synthesizes."""
    print("\n" + "=" * 60)
    print("Debate Collaboration")
    print("=" * 60)

    # Pro position
    print("\n[Pro Agent]")
    pro = complete("claude-sonnet-4-20250514", [
        user(f"Argue in favor of: {topic}")
    ], system="You argue FOR positions. Be persuasive but factual.").content
    print(f"Pro: {pro[:150]}...")

    # Con position
    print("\n[Con Agent]")
    con = complete("claude-sonnet-4-20250514", [
        user(f"Argue against: {topic}")
    ], system="You argue AGAINST positions. Be persuasive but factual.").content
    print(f"Con: {con[:150]}...")

    # Synthesizer
    print("\n[Synthesizer]")
    synthesis = complete("claude-sonnet-4-20250514", [
        user(f"Synthesize these opposing views into a balanced conclusion:\n\nPro: {pro}\n\nCon: {con}")
    ], system="You synthesize opposing views into balanced, nuanced conclusions.").content
    print(f"Synthesis: {synthesis[:150]}...")

    return synthesis


async def parallel_specialists(question: str) -> str:
    """Multiple specialists work in parallel, results combined."""
    print("\n" + "=" * 60)
    print("Parallel Specialists")
    print("=" * 60)

    specialists = {
        "technical": "You are a technical expert. Focus on how things work.",
        "practical": "You are practical-minded. Focus on real-world applications.",
        "historical": "You are a historian. Focus on origins and evolution.",
    }

    # Run all specialists in parallel
    async def get_perspective(name: str, system: str) -> tuple[str, str]:
        response = await acomplete(
            "claude-sonnet-4-20250514",
            [user(f"Answer from your perspective: {question}")],
            system=system,
        )
        return name, response.content

    tasks = [get_perspective(name, system) for name, system in specialists.items()]
    results = await asyncio.gather(*tasks)

    print("\nPerspectives gathered:")
    for name, content in results:
        print(f"  [{name}]: {content[:80]}...")

    # Combine perspectives
    combined = "\n\n".join(f"[{name}]: {content}" for name, content in results)
    final = await acomplete("claude-sonnet-4-20250514", [
        user(f"Combine these expert perspectives into a comprehensive answer:\n\n{combined}")
    ])

    print(f"\nCombined: {final.content[:200]}...")
    return final.content


def hierarchical_agents(task: str) -> str:
    """Manager delegates to workers, reviews results."""
    print("\n" + "=" * 60)
    print("Hierarchical Agents")
    print("=" * 60)

    # Manager breaks down task
    print("\n[Manager]")
    breakdown = complete("claude-sonnet-4-20250514", [
        user(f"Break this into 2-3 subtasks for your team:\n\n{task}")
    ], system="You are a manager. Delegate effectively and review work.").content
    print(f"Breakdown:\n{breakdown}")

    # Workers execute (simplified: one worker handles all)
    print("\n[Worker]")
    work = complete("claude-sonnet-4-20250514", [
        user(f"Complete these assigned tasks:\n\n{breakdown}")
    ], system="You are a diligent worker. Complete tasks thoroughly.").content
    print(f"Work completed: {work[:200]}...")

    # Manager reviews
    print("\n[Manager Review]")
    review = complete("claude-sonnet-4-20250514", [
        user(f"Review this work and provide final output:\n\nOriginal task: {task}\n\nCompleted work:\n{work}")
    ], system="You are a manager. Review work quality and compile final results.").content
    print(f"Final review: {review[:200]}...")

    return review


if __name__ == "__main__":
    pipeline_collaboration("the impact of AI on employment")
    debate_collaboration("Remote work is better than office work")
    asyncio.run(parallel_specialists("What is machine learning?"))
    hierarchical_agents("Create a marketing plan for a new mobile app")
