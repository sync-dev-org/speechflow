"""Command checks for the procedures in core-owned documentation."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]

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


def _documentation_workflow() -> dict[str, Any]:
    source = (REPOSITORY_ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    return cast(dict[str, Any], yaml.load(source, Loader=yaml.BaseLoader))


@pytest.mark.req("SP-SF-070")
def test_documentation_workflow_builds_public_main_and_deploys_pages() -> None:
    """Public main pushes and manual runs build locked documentation and deploy its artifact to Pages."""
    workflow = _documentation_workflow()
    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    assert workflow["on"]["push"] == {"branches": ["main"]}
    jobs = workflow["jobs"]
    assert set(jobs) == {"build", "deploy"}
    public_repository = "github.repository == 'sync-dev-org/speechflow'"
    assert jobs["build"]["if"] == public_repository
    assert jobs["deploy"]["if"] == public_repository + " && github.ref == 'refs/heads/main'"
    assert jobs["deploy"]["needs"] == "build"

    build_steps = jobs["build"]["steps"]
    setup_uv = next(step for step in build_steps if step.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", setup_uv["with"]["version"])
    assert setup_uv["with"]["python-version"] == "3.12"
    assert [shlex.split(step["run"]) for step in build_steps if "run" in step] == [
        ["uv", "sync", "--locked", "--only-group", "docs", "--no-install-project"],
        ["uv", "run", "--no-sync", "mkdocs", "build", "--strict"],
    ]
    upload = next(step for step in build_steps if step.get("uses", "").startswith("actions/upload-pages-artifact@"))
    assert upload["with"]["path"] == "site"
    deployment = jobs["deploy"]["steps"][0]
    assert deployment["uses"].startswith("actions/deploy-pages@")
    assert jobs["deploy"]["environment"] == {
        "name": "github-pages",
        "url": "${{ steps." + deployment["id"] + ".outputs.page_url }}",
    }


@pytest.mark.req("SP-SF-071")
def test_documentation_workflow_limits_permissions_and_operations_to_pages() -> None:
    """Only deployment can write Pages and request an ID token, without secrets or non-documentation steps."""
    workflow = _documentation_workflow()
    assert "secrets." not in (REPOSITORY_ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert workflow["permissions"] == {}
    jobs = workflow["jobs"]
    assert jobs["build"]["permissions"] == {"contents": "read"}
    assert jobs["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}
    assert [step["uses"].split("@")[0] for step in jobs["build"]["steps"] if "uses" in step] == [
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/upload-pages-artifact",
    ]
    assert len(jobs["deploy"]["steps"]) == 1
    assert "run" not in jobs["deploy"]["steps"][0]
