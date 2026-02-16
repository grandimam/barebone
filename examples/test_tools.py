from typing import Any

from barebone import Tool
from barebone import complete


MODEL = "claude-sonnet-4-20250514"


class AddTool(Tool):
    """Add two numbers, and return the result"""
    arg_one: int | None = None
    arg_two: int | None = None

    def execute(self) -> Any:
        return self.arg_one + self.arg_two


if __name__ == "__main__":
    complete()