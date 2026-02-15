from barebone import complete, user


def chain(model: str, *prompts):
    result = ""
    for prompt in prompts:
        if callable(prompt):
            prompt = prompt(result)
        response = complete(model, [user(prompt)])
        result = response.content
    return result


def basic_chaining():
    print("=" * 60)
    print("Basic Prompt Chaining")
    print("=" * 60)

    # Step 1: Generate outline
    response = complete("claude-sonnet-4-20250514", [
        user("Create a 3-point outline for an article about AI safety")
    ])
    outline = response.content
    print(f"Outline:\n{outline}\n")

    # Step 2: Expand into article
    response = complete("claude-sonnet-4-20250514", [
        user(f"Expand this outline into a short article (2-3 paragraphs):\n\n{outline}")
    ])
    article = response.content
    print(f"Article:\n{article}\n")

    # Step 3: Generate title
    response = complete("claude-sonnet-4-20250514", [
        user(f"Write a compelling title for this article:\n\n{article}")
    ])
    title = response.content
    print(f"Title: {title}")


def chain_helper():
    """Using the chain helper function."""
    print("\n" + "=" * 60)
    print("Chain Helper Function")
    print("=" * 60)

    result = chain(
        "claude-sonnet-4-20250514",
        "List 3 startup ideas in fintech. Just the names and one sentence each.",
        lambda ideas: f"Pick the best idea and explain why in 2 sentences:\n{ideas}",
        lambda pick: f"Write a one-paragraph elevator pitch:\n{pick}",
    )
    print(result)


def parallel_then_merge():
    """Run parallel chains then merge results."""
    print("\n" + "=" * 60)
    print("Parallel Chains + Merge")
    print("=" * 60)

    topic = "remote work"

    # Parallel: Get different perspectives
    pros = complete("claude-sonnet-4-20250514", [
        user(f"List 3 pros of {topic}. Be brief.")
    ]).content

    cons = complete("claude-sonnet-4-20250514", [
        user(f"List 3 cons of {topic}. Be brief.")
    ]).content

    # Merge: Synthesize perspectives
    response = complete("claude-sonnet-4-20250514", [
        user(f"Given these pros and cons, write a balanced 2-sentence summary:\n\nPros:\n{pros}\n\nCons:\n{cons}")
    ])

    print(f"Pros:\n{pros}\n")
    print(f"Cons:\n{cons}\n")
    print(f"Summary:\n{response.content}")


def iterative_refinement():
    """Iteratively refine output."""
    print("\n" + "=" * 60)
    print("Iterative Refinement")
    print("=" * 60)

    # Initial draft
    draft = complete("claude-sonnet-4-20250514", [
        user("Write a haiku about programming.")
    ]).content
    print(f"Draft 1:\n{draft}\n")

    # Refine
    for i in range(2):
        draft = complete("claude-sonnet-4-20250514", [
            user(f"Improve this haiku. Make it more evocative:\n\n{draft}")
        ]).content
        print(f"Draft {i + 2}:\n{draft}\n")


if __name__ == "__main__":
    basic_chaining()
    chain_helper()
    parallel_then_merge()
    iterative_refinement()
