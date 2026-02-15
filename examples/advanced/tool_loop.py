"""
Tool Loop pattern.

LLM calls tools until task is complete. This is what Agent does internally.
"""

from barebone import complete, execute, user, assistant, tool_result, Tool, Param


class Calculator(Tool):
    """Perform arithmetic calculations."""
    expression: str = Param(description="Math expression to evaluate")

    def execute(self) -> str:
        try:
            result = eval(self.expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class GetFact(Tool):
    """Get a fact about a number."""
    number: int = Param(description="The number to get a fact about")

    def execute(self) -> str:
        facts = {
            42: "The answer to life, the universe, and everything",
            7: "Considered lucky in many cultures",
            13: "Considered unlucky in Western culture",
        }
        return facts.get(self.number, f"{self.number} is just a number")


def tool_loop(query: str, tools: list, max_turns: int = 5) -> str:
    """Run tool loop until LLM stops calling tools."""
    print("=" * 60)
    print("Tool Loop")
    print("=" * 60)

    messages = [user(query)]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response = complete("claude-sonnet-4-20250514", messages, tools=tools)

        # No tool calls = done
        if not response.tool_calls:
            print(f"Final response: {response.content}")
            return response.content

        # Process tool calls
        print(f"Tool calls: {[tc.name for tc in response.tool_calls]}")

        # Add assistant message with tool use
        messages.append(assistant(response.content or ""))

        for tc in response.tool_calls:
            result = execute(tc, tools)
            print(f"  {tc.name}({tc.arguments}) -> {result}")
            messages.append(tool_result(tc, result))

    return "Max turns reached"


def multi_tool_example():
    """Example with multiple tools."""
    print("\n" + "=" * 60)
    print("Multi-Tool Example")
    print("=" * 60)

    tools = [Calculator, GetFact]

    result = tool_loop(
        "What is 6 * 7, and can you tell me an interesting fact about that number?",
        tools
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    tools = [Calculator]
    tool_loop("What is (15 + 27) * 3?", tools)
    multi_tool_example()
