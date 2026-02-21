import os

from barebone import complete
from barebone import user

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"


def chain(*prompts):
    result = ""
    for prompt in prompts:
        if callable(prompt):
            prompt = prompt(result)
        response = complete(MODEL, [user(prompt)], api_key=API_KEY)
        result = response.content
    return result


def basic_chaining():
    print("=" * 60)
    print("Basic Prompt Chaining")
    print("=" * 60)

    response = complete(
        MODEL, [user("Create a 3-point outline for an article about AI safety")], api_key=API_KEY
    )
    outline = response.content
    print(f"Outline:\n{outline}\n")

    response = complete(
        MODEL, [user(f"Expand this outline into a short article:\n\n{outline}")], api_key=API_KEY
    )
    article = response.content
    print(f"Article:\n{article}\n")

    response = complete(
        MODEL, [user(f"Write a compelling title for this article:\n\n{article}")], api_key=API_KEY
    )
    print(f"Title: {response.content}")


def chain_helper():
    print("\n" + "=" * 60)
    print("Chain Helper Function")
    print("=" * 60)

    result = chain(
        "List 3 startup ideas in fintech. Just the names.",
        lambda ideas: f"Pick the best idea and explain why:\n{ideas}",
        lambda pick: f"Write a one-paragraph pitch:\n{pick}",
    )
    print(result)


if __name__ == "__main__":
    basic_chaining()
    chain_helper()
