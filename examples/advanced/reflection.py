"""
Reflection pattern.

LLM critiques and improves its own output.
"""

from barebone import complete, user


def self_reflection(task: str, max_reflections: int = 2) -> str:
    """Generate output, reflect on it, improve."""
    print("=" * 60)
    print("Self Reflection")
    print("=" * 60)

    # Initial attempt
    response = complete("claude-sonnet-4-20250514", [
        user(task)
    ])
    output = response.content
    print(f"Initial:\n{output}\n")

    for i in range(max_reflections):
        # Reflect
        reflection = complete("claude-sonnet-4-20250514", [
            user(f"""Reflect on your previous response to: "{task}"

Your response was:
{output}

What could be improved? Be specific. If it's already excellent, say "NO IMPROVEMENTS NEEDED".""")
        ]).content

        print(f"Reflection {i + 1}:\n{reflection}\n")

        if "NO IMPROVEMENTS NEEDED" in reflection.upper():
            break

        # Improve based on reflection
        output = complete("claude-sonnet-4-20250514", [
            user(f"""Improve your response based on your reflection.

Original task: {task}

Your previous response:
{output}

Your reflection:
{reflection}

Improved response:""")
        ]).content

        print(f"Improved:\n{output}\n")

    return output


def chain_of_thought_reflection(question: str) -> str:
    """Reflect on reasoning process, not just output."""
    print("\n" + "=" * 60)
    print("Chain-of-Thought Reflection")
    print("=" * 60)

    # Initial reasoning
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Think through this step by step:

{question}

Show your reasoning, then give your answer.""")
    ])
    reasoning = response.content
    print(f"Initial reasoning:\n{reasoning}\n")

    # Reflect on reasoning
    reflection = complete("claude-sonnet-4-20250514", [
        user(f"""Review your reasoning for any logical errors or gaps:

{reasoning}

List any issues with the reasoning process. If none, say "REASONING VALID".""")
    ]).content

    print(f"Reflection on reasoning:\n{reflection}\n")

    if "REASONING VALID" not in reflection.upper():
        # Re-reason with corrections
        response = complete("claude-sonnet-4-20250514", [
            user(f"""Redo your reasoning, addressing these issues:

{reflection}

Original question: {question}

Corrected reasoning and answer:""")
        ])
        reasoning = response.content
        print(f"Corrected reasoning:\n{reasoning}\n")

    return reasoning


def debate_reflection(topic: str) -> str:
    """Generate opposing viewpoints, then synthesize."""
    print("\n" + "=" * 60)
    print("Debate Reflection")
    print("=" * 60)

    # Position A
    position_a = complete("claude-sonnet-4-20250514", [
        user(f"Argue strongly FOR: {topic}")
    ]).content
    print(f"For:\n{position_a}\n")

    # Position B
    position_b = complete("claude-sonnet-4-20250514", [
        user(f"Argue strongly AGAINST: {topic}")
    ]).content
    print(f"Against:\n{position_b}\n")

    # Synthesize
    synthesis = complete("claude-sonnet-4-20250514", [
        user(f"""Given these two positions, provide a balanced, nuanced view:

FOR: {position_a}

AGAINST: {position_b}

Synthesized view:""")
    ]).content

    print(f"Synthesis:\n{synthesis}")
    return synthesis


if __name__ == "__main__":
    self_reflection("Explain recursion to a beginner in 2-3 sentences")
    chain_of_thought_reflection("If a train leaves at 3pm going 60mph, and another leaves at 4pm going 80mph from 100 miles behind, when do they meet?")
    debate_reflection("AI will create more jobs than it eliminates")
