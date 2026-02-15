from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from typing import Any

from .base import Tool, Param


class Python(Tool):
    """Execute Python code and return the output. Runs in sandboxed subprocess by default."""

    code: str = Param(description="Python code to execute")
    timeout: int = Param(default=60, description="Maximum execution time in seconds")
    sandbox: bool = Param(default=True, description="Run in isolated subprocess")

    async def execute(self) -> str:
        if self.sandbox:
            return await self._run_sandboxed()
        else:
            return await self._run_inline()

    async def _run_sandboxed(self) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            escaped_code = self.code.replace('"""', "'''").replace(chr(92), chr(92) + chr(92))
            wrapped_code = f'''
                import sys
                import json
                
                _output = []
                
                try:
                    exec(compile("""{escaped_code}""", "<code>", "exec"), globals())
                except Exception as e:
                    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
                    sys.exit(1)
            '''
            f.write(wrapped_code)
            temp_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Execution timed out after {self.timeout} seconds"

            output_parts = []

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            if stdout_text:
                output_parts.append(stdout_text)

            if stderr_text:
                output_parts.append(f"Stderr:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"Exit code: {process.returncode}")

            return "\n".join(output_parts) if output_parts else "(no output)"

        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    async def _run_inline(self) -> str:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }

        result = None

        def execute_code():
            nonlocal result
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    try:
                        compiled = compile(self.code, "<code>", "eval")
                        result = eval(compiled, namespace)
                    except SyntaxError:
                        compiled = compile(self.code, "<code>", "exec")
                        exec(compiled, namespace)
                except Exception:
                    traceback.print_exc()

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, execute_code),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return f"Error: Execution timed out after {self.timeout} seconds"

        output_parts = []

        stdout_text = stdout_capture.getvalue().strip()
        stderr_text = stderr_capture.getvalue().strip()

        if stdout_text:
            output_parts.append(stdout_text)

        if result is not None:
            output_parts.append(f"Result: {result!r}")

        if stderr_text:
            output_parts.append(f"Stderr:\n{stderr_text}")

        return "\n".join(output_parts) if output_parts else "(no output)"


__all__ = [
    "Python",
]
