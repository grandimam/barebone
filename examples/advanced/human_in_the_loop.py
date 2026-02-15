"""
Human-in-the-Loop pattern.

Pause for human approval or input at critical points.
"""

from barebone import complete, execute, user, tool_result, Tool, Param


class SendEmail(Tool):
    """Send an email (requires approval)."""
    to: str = Param(description="Recipient email")
    subject: str = Param(description="Email subject")
    body: str = Param(description="Email body")

    def execute(self) -> str:
        return f"Email sent to {self.to}"


class DeleteFile(Tool):
    """Delete a file (requires approval)."""
    path: str = Param(description="File path to delete")

    def execute(self) -> str:
        return f"Deleted {self.path}"


# Tools that require human approval
REQUIRES_APPROVAL = {"SendEmail", "DeleteFile"}


def get_human_approval(tool_call) -> tuple[bool, str]:
    """Get human approval for a tool call."""
    print(f"\n{'='*40}")
    print(f"APPROVAL REQUIRED: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"{'='*40}")

    response = input("Approve? (y/n/modify): ").strip().lower()

    if response == "y":
        return True, ""
    elif response == "n":
        reason = input("Reason for rejection: ")
        return False, reason
    else:
        # Allow modification
        print("Enter modified arguments as key=value, empty to finish:")
        modified = dict(tool_call.arguments)
        while True:
            mod = input("> ").strip()
            if not mod:
                break
            key, value = mod.split("=", 1)
            modified[key.strip()] = value.strip()
        tool_call.arguments = modified
        return True, ""


def human_approval_loop(query: str, tools: list) -> str:
    """Tool loop with human approval for sensitive operations."""
    print("=" * 60)
    print("Human-in-the-Loop")
    print("=" * 60)

    messages = [user(query)]

    while True:
        response = complete("claude-sonnet-4-20250514", messages, tools=tools)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            # Check if approval needed
            if tc.name in REQUIRES_APPROVAL:
                approved, reason = get_human_approval(tc)
                if not approved:
                    result = f"REJECTED: {reason}"
                    messages.append(tool_result(tc, result))
                    continue

            result = execute(tc, tools)
            print(f"Executed {tc.name}: {result}")
            messages.append(tool_result(tc, result))


def human_feedback_loop(task: str) -> str:
    """Iterative refinement with human feedback."""
    print("\n" + "=" * 60)
    print("Human Feedback Loop")
    print("=" * 60)

    output = complete("claude-sonnet-4-20250514", [user(task)]).content

    while True:
        print(f"\nCurrent output:\n{output}\n")
        feedback = input("Feedback (or 'done'): ").strip()

        if feedback.lower() == "done":
            break

        output = complete("claude-sonnet-4-20250514", [
            user(f"""Revise based on feedback.

Original task: {task}

Current output:
{output}

Feedback: {feedback}

Revised:""")
        ]).content

    return output


def human_choice_branch(query: str) -> str:
    """Let human choose between AI-generated options."""
    print("\n" + "=" * 60)
    print("Human Choice Branch")
    print("=" * 60)

    # Generate options
    response = complete("claude-sonnet-4-20250514", [
        user(f"""For this request, provide 3 different approaches.
Number them 1, 2, 3.

Request: {query}""")
    ])

    print(f"Options:\n{response.content}\n")

    choice = input("Choose option (1/2/3): ").strip()

    # Execute chosen option
    response = complete("claude-sonnet-4-20250514", [
        user(f"""Execute option {choice} for: {query}

The options were:
{response.content}

Now implement option {choice} in detail:""")
    ])

    return response.content


if __name__ == "__main__":
    # These examples require interactive input
    print("Example 1: Human Feedback Loop")
    print("-" * 40)
    result = human_feedback_loop("Write a product description for a smart water bottle")
    print(f"\nFinal: {result}")

    print("\n\nExample 2: Human Choice Branch")
    print("-" * 40)
    result = human_choice_branch("Help me plan a weekend trip")
    print(f"\nResult: {result}")
