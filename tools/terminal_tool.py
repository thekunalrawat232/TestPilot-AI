"""Terminal / shell command execution tool."""

from __future__ import annotations

import subprocess
from typing import Any

from langchain_core.tools import tool

# Commands that are never allowed to run
_BLOCKED_PATTERNS = {"rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"}


@tool
def run_shell_command(command: str, timeout_seconds: int = 60) -> dict[str, Any]:
    """Run an arbitrary shell command and return stdout/stderr.

    Safety: destructive root-level commands are blocked.

    Args:
        command: Shell command string to execute.
        timeout_seconds: Max seconds before the command is killed (default 60).
    """
    lowered = command.lower()
    for pat in _BLOCKED_PATTERNS:
        if pat in lowered:
            return {"success": False, "error": f"Blocked dangerous command pattern: {pat}"}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout[-8000:] if result.stdout else "",
            "stderr": result.stderr[-4000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout_seconds}s"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
                