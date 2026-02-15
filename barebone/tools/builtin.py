from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from .base import Tool, Param


class Read(Tool):
    """Read a file from the filesystem. Returns contents with line numbers."""

    file_path: str = Param(description="Absolute path to the file to read")
    offset: int = Param(default=0, description="Line number to start reading from (0-based)")
    limit: int = Param(default=2000, description="Maximum number of lines to read")

    def execute(self) -> str:
        path = Path(self.file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {self.file_path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        selected = lines[self.offset : self.offset + self.limit]

        result = []
        for i, line in enumerate(selected, start=self.offset + 1):
            result.append(f"{i:6d}\t{line.rstrip()}")

        return "\n".join(result)


class Write(Tool):
    """Write content to a file. Creates parent directories if needed."""

    file_path: str = Param(description="Absolute path to the file to write")
    content: str = Param(description="Content to write to the file")

    def execute(self) -> str:
        path = Path(self.file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.content)

        return f"Successfully wrote {len(self.content)} bytes to {self.file_path}"


class Edit(Tool):
    """Edit a file by replacing a unique string with new content."""

    file_path: str = Param(description="Absolute path to the file to edit")
    old_string: str = Param(description="The exact string to find and replace")
    new_string: str = Param(description="The replacement string")

    def execute(self) -> str:
        path = Path(self.file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = path.read_text(encoding="utf-8")

        if self.old_string not in content:
            raise ValueError(f"String not found in file: {self.old_string[:50]}...")

        count = content.count(self.old_string)
        if count > 1:
            raise ValueError(
                f"String appears {count} times. Provide more context to make it unique."
            )

        new_content = content.replace(self.old_string, self.new_string)
        path.write_text(new_content, encoding="utf-8")

        return f"Successfully edited {self.file_path}"


class Bash(Tool):
    """Execute a bash command and return the output."""

    command: str = Param(description="The bash command to execute")
    cwd: str | None = Param(default=None, description="Working directory for the command")
    timeout: int = Param(default=120, description="Timeout in seconds")

    async def execute(self) -> str:
        working_dir = self.cwd or os.getcwd()

        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=working_dir,
                env={**os.environ, "TERM": "dumb"},
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            output = stdout.decode("utf-8", errors="replace")

            if process.returncode != 0:
                return f"Exit code: {process.returncode}\n{output}"

            return output or "(no output)"

        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Command timed out after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Command failed: {e}")


class Glob(Tool):
    """Find files matching a glob pattern."""

    pattern: str = Param(description="Glob pattern (e.g., '**/*.py')")
    path: str | None = Param(default=None, description="Directory to search in")

    def execute(self) -> str:
        search_path = Path(self.path or os.getcwd()).expanduser().resolve()

        if not search_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")

        matches = sorted(search_path.glob(self.pattern))

        if not matches:
            return f"No files matching '{self.pattern}'"

        return "\n".join(str(m) for m in matches[:100])


class Grep(Tool):
    """Search for a regex pattern in files."""

    pattern: str = Param(description="Regex pattern to search for")
    path: str | None = Param(default=None, description="Directory to search in")
    glob: str = Param(default="**/*", description="Glob pattern to filter files")

    def execute(self) -> str:
        search_path = Path(self.path or os.getcwd()).expanduser().resolve()

        if not search_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")

        regex = re.compile(self.pattern)
        results = []

        for file_path in search_path.glob(self.glob):
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
            return f"No matches for '{self.pattern}'"

        return "\n".join(results)


__all__ = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
]
