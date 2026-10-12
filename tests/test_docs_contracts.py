"""Command checks for the procedures in core-owned documentation."""

from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

GATE_COMMANDS = (
    "uv sync",
    "uv lock --check",
    "uv run pytest",
    "uv run ruff check",
    "uv run ruff format --check",
    "uv run mypy",
    "uv build --wheel",
    "uv run mkdocs build --strict",
)
REQUIRED_COMMANDS = {
    "README.md": GATE_COMMANDS,
    "README.ja.md": GATE_COMMANDS,
}
RETIRED_COMMANDS = (
    "scripts/check_workspace.py",
    "scripts/release",
    "uv build --all-packages",
    "uv build --package speechflow",
    "uv publish",
    "pip install speechflow[",
)


@pytest.mark.parametrize("relative_path", REQUIRED_COMMANDS, ids=str)
def test_procedures_use_only_current_core_commands(relative_path: str) -> None:
    """Procedure documents list the current core verification and release commands and no retired command."""
    document = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")

    for command in REQUIRED_COMMANDS[relative_path]:
        assert command in document, command
    for command in RETIRED_COMMANDS:
        assert command not in document, command
