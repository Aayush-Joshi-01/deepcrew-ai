from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

from ...exceptions import SkillError
from ..base import Skill


class CodeExecutionSkill(Skill):
    """Execute Python code in a subprocess and return stdout/stderr."""

    name = "code_exec"
    description = "Execute a snippet of Python code and return the output."
    # Class-level default satisfying the Skill instance annotation (shared, never mutated)
    parameters: dict[str, Any] = {  # noqa: RUF012
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"},
            "language": {
                "type": "string",
                "description": "Programming language (only 'python' supported)",
                "default": "python",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def execute(self, code: str, language: str = "python", **_: Any) -> str:  # type: ignore[override]
        if language.lower() != "python":
            raise SkillError(f"CodeExecutionSkill only supports 'python', got {language!r}")

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired as exc:
            raise SkillError(f"Code execution timed out after {self._timeout}s") from exc
        except Exception as exc:
            raise SkillError(f"Code execution failed: {exc}") from exc
