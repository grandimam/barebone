import os

from barebone import complete, user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


def self_reflection(task: str, max_reflections: int = 2) -> str:
    print("=" * 60)
    print("Self Reflection")
    print("=" * 60)

    response = complete(MODEL, [user(task)], api_key=API_KEY)
    output = response.content
    print(f"Initial:\n{output}\n")

    for i in range(max_reflections):
        reflection = complete(MODEL, [
            user(f"""Reflect on your previous response to: "{task}"

Your response was:
{output}

What could be improved? Be specific. If it's already excellent, say "NO IMPROVEMENTS NEEDED".""")
        ], api_key=API_KEY).content

        print(f"Reflection {i + 1}:\n{reflection}\n")

        if "NO IMPROVEMENTS NEEDED" in reflection.upper():
            break

        output = complete(MODEL, [
            user(f"""Improve your response based on your reflection.

Original task: {task}

Your previous response:
{output}

Your reflection:
{reflection}

Improved response:""")
        ], api_key=API_KEY).content

        print(f"Improved:\n{output}\n")

    return output


def chain_of_thought_reflection(question: str) -> str:
    print("\n" + "=" * 60)
    print("Chain-of-Thought Reflection")
    print("=" * 60)

    response = complete(MODEL, [
        user(f"""Think through this step by step:

{question}

Show your reasoning, then give your answer.""")
    ], api_key=API_KEY)
    reasoning = response.content
    print(f"Initial reasoning:\n{reasoning}\n")

    reflection = complete(MODEL, [
        user(f"""Review your reasoning for any logical errors or gaps:

{reasoning}

List any issues with the reasoning process. If none, say "REASONING VALID".""")
    ], api_key=API_KEY).content

    print(f"Reflection on reasoning:\n{reflection}\n")

    if "REASONING VALID" not in reflection.upper():
        response = complete(MODEL, [
            user(f"""Redo your reasoning, addressing these issues:

{reflection}

Original question: {question}

Corrected reasoning and answer:""")
        ], api_key=API_KEY)
        reasoning = response.content
        print(f"Corrected reasoning:\n{reasoning}\n")

    return reasoning


def debate_reflection(topic: str) -> str:
    print("\n" + "=" * 60)
    print("Debate Reflection")
    print("=" * 60)

    position_a = complete(MODEL, [
        user(f"Argue strongly FOR: {topic}")
    ], api_key=API_KEY).content
    print(f"For:\n{position_a}\n")

    position_b = complete(MODEL, [
        user(f"Argue strongly AGAINST: {topic}")
    ], api_key=API_KEY).content
    print(f"Against:\n{position_b}\n")

    synthesis = complete(MODEL, [
        user(f"""Given these two positions, provide a balanced, nuanced view:

FOR: {position_a}

AGAINST: {position_b}

Synthesized view:""")
    ], api_key=API_KEY).content

    print(f"Synthesis:\n{synthesis}")
    return synthesis


if __name__ == "__main__":
    self_reflection("Explain recursion to a beginner in 2-3 sentences")
    chain_of_thought_reflection("If a train leaves at 3pm going 60mph, and another leaves at 4pm going 80mph from 100 miles behind, when do they meet?")
    debate_reflection("AI will create more jobs than it eliminates")
