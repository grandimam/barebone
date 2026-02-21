import asyncio
import os
from dataclasses import dataclass

from barebone import acomplete
from barebone import complete
from barebone import user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
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


def agent_call(agent_name: str, message: str) -> str:
    agent = AGENTS[agent_name]
    response = complete(MODEL, [user(message)], api_key=API_KEY, system=agent.system)
    return response.content


def pipeline_collaboration(topic: str) -> str:
    print("=" * 60)
    print("Pipeline Collaboration")
    print("=" * 60)

    print("\n[Researcher]")
    research = agent_call("researcher", f"Research key facts about: {topic}")
    print(f"Research: {research[:200]}...")

    print("\n[Writer]")
    draft = agent_call("writer", f"Write a short article using this research:\n\n{research}")
    print(f"Draft: {draft[:200]}...")

    print("\n[Editor]")
    final = agent_call("editor", f"Edit and polish this article:\n\n{draft}")
    print(f"Final: {final[:200]}...")

    return final


def debate_collaboration(topic: str) -> str:
    print("\n" + "=" * 60)
    print("Debate Collaboration")
    print("=" * 60)

    print("\n[Pro Agent]")
    pro = complete(
        MODEL,
        [user(f"Argue in favor of: {topic}")],
        api_key=API_KEY,
        system="You argue FOR positions. Be persuasive but factual.",
    ).content
    print(f"Pro: {pro[:150]}...")

    print("\n[Con Agent]")
    con = complete(
        MODEL,
        [user(f"Argue against: {topic}")],
        api_key=API_KEY,
        system="You argue AGAINST positions. Be persuasive but factual.",
    ).content
    print(f"Con: {con[:150]}...")

    print("\n[Synthesizer]")
    synthesis = complete(
        MODEL,
        [
            user(
                f"Synthesize these opposing views into a balanced conclusion:\n\nPro: {pro}\n\nCon: {con}"
            )
        ],
        api_key=API_KEY,
        system="You synthesize opposing views into balanced, nuanced conclusions.",
    ).content
    print(f"Synthesis: {synthesis[:150]}...")

    return synthesis


async def parallel_specialists(question: str) -> str:
    print("\n" + "=" * 60)
    print("Parallel Specialists")
    print("=" * 60)

    specialists = {
        "technical": "You are a technical expert. Focus on how things work.",
        "practical": "You are practical-minded. Focus on real-world applications.",
        "historical": "You are a historian. Focus on origins and evolution.",
    }

    async def get_perspective(name: str, system: str) -> tuple[str, str]:
        response = await acomplete(
            MODEL,
            [user(f"Answer from your perspective: {question}")],
            api_key=API_KEY,
            system=system,
        )
        return name, response.content

    tasks = [get_perspective(name, system) for name, system in specialists.items()]
    results = await asyncio.gather(*tasks)

    print("\nPerspectives gathered:")
    for name, content in results:
        print(f"  [{name}]: {content[:80]}...")

    combined = "\n\n".join(f"[{name}]: {content}" for name, content in results)
    final = await acomplete(
        MODEL,
        [user(f"Combine these expert perspectives into a comprehensive answer:\n\n{combined}")],
        api_key=API_KEY,
    )

    print(f"\nCombined: {final.content[:200]}...")
    return final.content


def hierarchical_agents(task: str) -> str:
    print("\n" + "=" * 60)
    print("Hierarchical Agents")
    print("=" * 60)

    print("\n[Manager]")
    breakdown = complete(
        MODEL,
        [user(f"Break this into 2-3 subtasks for your team:\n\n{task}")],
        api_key=API_KEY,
        system="You are a manager. Delegate effectively and review work.",
    ).content
    print(f"Breakdown:\n{breakdown}")

    print("\n[Worker]")
    work = complete(
        MODEL,
        [user(f"Complete these assigned tasks:\n\n{breakdown}")],
        api_key=API_KEY,
        system="You are a diligent worker. Complete tasks thoroughly.",
    ).content
    print(f"Work completed: {work[:200]}...")

    print("\n[Manager Review]")
    review = complete(
        MODEL,
        [
            user(
                f"Review this work and provide final output:\n\nOriginal task: {task}\n\nCompleted work:\n{work}"
            )
        ],
        api_key=API_KEY,
        system="You are a manager. Review work quality and compile final results.",
    ).content
    print(f"Final review: {review[:200]}...")

    return review


if __name__ == "__main__":
    pipeline_collaboration("the impact of AI on employment")
    debate_collaboration("Remote work is better than office work")
    asyncio.run(parallel_specialists("What is machine learning?"))
    hierarchical_agents("Create a marketing plan for a new mobile app")
