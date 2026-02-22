from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from barebone.tools.base import tool
from barebone.types import NullableStr
from barebone.types import QuestionOption
from barebone.types import Question


def _parse_response(response: str, options: list[QuestionOption], multi_select: bool) -> str:
    response = response.strip()
    if not response:
        return "(no response)"

    try:
        if multi_select:
            indices = [int(s.strip()) - 1 for s in response.split(",")]
        else:
            indices = [int(response) - 1]

        labels = [options[i].label for i in indices if 0 <= i < len(options)]
        return ", ".join(labels) if labels else response
    except ValueError:
        return response


@tool
def ask_user_question(questions: list[Question]) -> dict[str, Any]:
    """Ask the user clarifying questions with multiple-choice options.

    Use this tool when you need clarification, additional information,
    or confirmation from the user before proceeding with a task.
    """
    answers: dict[str, str] = {}

    for q in questions:
        print(f"\n{q.header}: {q.question}")

        for i, opt in enumerate(q.options, 1):
            print(f"  {i}. {opt.label} - {opt.description}")

        if q.multiSelect:
            print("  (Enter numbers separated by commas, or type your own answer)")
        else:
            print("  (Enter a number, or type your own answer)")

        response = input("> ").strip()
        answers[q.question] = _parse_response(response, q.options, q.multiSelect)

    return {
        "questions": [q.model_dump() for q in questions],
        "answers": answers,
    }


@tool
def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file from the filesystem. Returns contents with line numbers."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    selected = lines[offset : offset + limit]

    result = []
    for i, line in enumerate(selected, start=offset + 1):
        result.append(f"{i:6d}\t{line.rstrip()}")

    return "\n".join(result)


@tool
def write(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    path = Path(file_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Successfully wrote {len(content)} bytes to {file_path}"


@tool
def edit(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing a unique string with new content."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8")

    if old_string not in content:
        raise ValueError(f"String not found in file: {old_string[:50]}...")

    count = content.count(old_string)
    if count > 1:
        raise ValueError(f"String appears {count} times. Provide more context to make it unique.")

    new_content = content.replace(old_string, new_string)
    path.write_text(new_content, encoding="utf-8")

    return f"Successfully edited {file_path}"


@tool
async def bash(command: str, cwd: NullableStr = None, timeout: int = 120) -> str:
    """Execute a bash command and return the output."""
    working_dir = cwd or os.getcwd()

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=working_dir,
            env={**os.environ, "TERM": "dumb"},
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")

        if process.returncode != 0:
            return f"Exit code: {process.returncode}\n{output}"

        return output or "(no output)"

    except TimeoutError:
        process.kill()
        raise TimeoutError(f"Command timed out after {timeout}s")
    except Exception as e:
        raise RuntimeError(f"Command failed: {e}")


@tool
def glob(pattern: str, path: NullableStr = None) -> str:
    """Find files matching a glob pattern."""
    search_path = Path(path or os.getcwd()).expanduser().resolve()

    if not search_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    matches = sorted(search_path.glob(pattern))

    if not matches:
        return f"No files matching '{pattern}'"

    return "\n".join(str(m) for m in matches[:100])


@tool
def grep(pattern: str, path: NullableStr = None, file_glob: str = "**/*") -> str:
    """Search for a regex pattern in files."""
    search_path = Path(path or os.getcwd()).expanduser().resolve()

    if not search_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    regex = re.compile(pattern)
    results = []

    for file_path in search_path.glob(file_glob):
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{file_path}:{i}: {line.strip()}")

                    if len(results) >= 100:
                        results.append("... (truncated)")
                        return "\n".join(results)
        except Exception:
            continue

    if not results:
        return f"No matches for '{pattern}'"

    return "\n".join(results)
