"""Standalone repository and wheel contracts for the SpeechFlow core."""

from __future__ import annotations

import email
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_EXPORTS = {
    "AudioData",
    "TTSEngineBase",
    "EngineCapabilities",
    "EngineState",
    "HealthSnapshot",
    "TTSError",
    "ConfigurationError",
    "EngineStateError",
    "EngineClosedError",
    "InferenceError",
    "EmptyOutputError",
    "EnvironmentUnavailableError",
    "ModuleNotInstalledError",
    "run_sync",
    "iterate_sync",
    "iterate_async",
    "__version__",
}


def _config() -> dict[str, Any]:
    return tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _run(*command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise AssertionError(f"invalid requirement: {requirement}")
    return match.group(0).lower().replace("_", "-")


def _requirement_signature(requirement: str) -> tuple[str, frozenset[str]]:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise AssertionError(f"invalid requirement: {requirement}")
    name = match.group(0).lower().replace("_", "-")
    specifiers = frozenset(part for part in requirement[match.end() :].replace(" ", "").split(",") if part)
    return name, specifiers


def _wheel_metadata(wheel: Path) -> tuple[email.message.Message, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        files = set(archive.namelist())
        metadata_path = next(path for path in files if path.endswith(".dist-info/METADATA"))
        metadata = email.message_from_string(archive.read(metadata_path).decode("utf-8"))
    return metadata, files


def _run_isolated(script: str, *, python: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    isolated_script = textwrap.dedent(script)
    if python is None:
        isolated_script = f"sys.path.insert(0, {str(REPOSITORY_ROOT / 'src')!r})\n" + isolated_script
    return subprocess.run(
        [str(python or sys.executable), "-I", "-c", "import sys\n" + isolated_script],
        cwd=REPOSITORY_ROOT.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    wheelhouse = tmp_path_factory.mktemp("core-wheel")
    _run("uv", "build", "--wheel", "--out-dir", str(wheelhouse), cwd=REPOSITORY_ROOT)
    return next(wheelhouse.glob("speechflow-*.whl"))


@pytest.fixture(scope="session")
def built_sdist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sdist_directory = tmp_path_factory.mktemp("core-sdist")
    _run("uv", "build", "--sdist", "--out-dir", str(sdist_directory), cwd=REPOSITORY_ROOT)
    return next(sdist_directory.glob("speechflow-*.tar.gz"))


@pytest.fixture(scope="session")
def installed_core_python(built_wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    environment = tmp_path_factory.mktemp("installed-core") / ".venv"
    _run("uv", "venv", "--python", sys.executable, str(environment), cwd=REPOSITORY_ROOT)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run("uv", "pip", "install", "--python", str(python), str(built_wheel), cwd=REPOSITORY_ROOT)
    return python


@pytest.mark.req("SP-SF-047")
@pytest.mark.req("SP-SF-049")
def test_sdist_contains_only_package_source_and_public_build_files(built_sdist: Path) -> None:
    """The source distribution contains exactly package source, README, license, and build metadata."""
    with tarfile.open(built_sdist) as archive:
        files = {member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()}
    source_files = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "src/speechflow").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }
    assert files == source_files | {"README.md", "LICENSE", "pyproject.toml", "PKG-INFO"}


@pytest.mark.req("SP-SF-047")
@pytest.mark.req("SP-SF-048")
@pytest.mark.req("SP-SF-049")
def test_sdist_builds_core_wheel_without_development_hook(built_sdist: Path, built_wheel: Path, tmp_path: Path) -> None:
    """The published source archive builds the same core wheel without the development build hook."""
    with tarfile.open(built_sdist) as archive:
        archive.extractall(tmp_path, filter="data")
    source_directory = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not (source_directory / "hatch_build.py").exists()
    wheelhouse = tmp_path / "rebuilt-wheel"
    _run("uv", "build", "--wheel", "--out-dir", str(wheelhouse), cwd=source_directory)
    rebuilt_wheel = next(wheelhouse.glob("speechflow-*.whl"))
    _, files = _wheel_metadata(rebuilt_wheel)
    _, expected_files = _wheel_metadata(built_wheel)
    assert files == expected_files


@pytest.mark.req("SP-SF-047")
@pytest.mark.req("SP-SF-048")
def test_wheel_metadata_publishes_readme_and_public_repository(built_wheel: Path) -> None:
    """The wheel description contains the README and links to the public repository."""
    metadata, _ = _wheel_metadata(built_wheel)
    assert metadata["Description-Content-Type"] == "text/markdown"
    description = metadata.get_payload()
    assert isinstance(description, str)
    assert description.strip() == (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8").strip()
    assert metadata.get_all("Project-URL", []) == ["Repository, https://github.com/sync-dev-org/speechflow"]


@pytest.mark.req("SP-SF-028")
@pytest.mark.req("SP-SF-047")
def test_root_metadata_declares_standalone_core_distribution() -> None:
    """Core metadata declares its distribution, Python range, dependencies, and build settings."""
    config = _config()
    project = config["project"]

    assert project["name"] == "speechflow"
    assert project["dynamic"] == ["version"]
    assert project["requires-python"] == ">=3.12,<3.14"
    assert project["dependencies"] == ["numpy>=1.26.4"]
    assert "optional-dependencies" not in project
    assert config["build-system"] == {
        "requires": ["hatchling", "hatch-vcs"],
        "build-backend": "hatchling.build",
    }

    dev_names = {_requirement_name(requirement) for requirement in config["dependency-groups"]["dev"]}
    assert dev_names == {
        "pytest",
        "pytest-asyncio",
        "pytest-mock",
        "pytest-cov",
        "ruff",
        "mypy",
        "noren",
        "mkdocs-material",
        "mkdocstrings",
    }

    uv_config = config.get("tool", {}).get("uv", {})
    assert "workspace" not in uv_config
    assert "sources" not in uv_config


def test_repository_tools_target_only_root_source_and_tests() -> None:
    """Core verification settings target the root source and tests."""
    config = _config()

    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert config["tool"]["ruff"]["src"] == ["src", "tests"]
    assert config["tool"]["mypy"]["mypy_path"] == ["src"]
    assert config["tool"]["mypy"]["files"] == ["src", "tests"]
    assert config["tool"]["coverage"]["run"]["source"] == ["src/speechflow"]


@pytest.mark.req("SP-SF-028")
def test_core_uses_standalone_root_layout() -> None:
    """The core uses its root source layout and tag based version settings."""
    source_root = REPOSITORY_ROOT / "src/speechflow"

    assert (source_root / "__init__.py").is_file()
    assert (source_root / "py.typed").is_file()
    assert not (REPOSITORY_ROOT / "packages").exists()
    assert not (REPOSITORY_ROOT / ".gitmodules").exists()

    config = _config()
    raw_options = config["tool"]["hatch"]["version"]["raw-options"]
    assert raw_options["root"] == "."
    assert raw_options["relative_to"] == "pyproject.toml"
    assert raw_options["fallback_version"] == "0.16.0"
    assert raw_options["tag_regex"] == r"^v(?P<version>[^/]+)$"
    assert raw_options["git_describe_command"][-1] == "v*"
    assert config["tool"]["hatch"]["build"]["hooks"]["vcs"]["version-file"] == "src/speechflow/_version.py"
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/speechflow"]


def test_gitignore_owns_only_the_root_generated_version_file() -> None:
    """Git ignores the root generated version file without module clone entries."""
    ignored = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "src/speechflow/_version.py" in ignored
    assert "packages/speechflow/src/speechflow/_version.py" not in ignored
    assert not any(rule.startswith("packages/") for rule in ignored)


@pytest.mark.req("SP-SF-047")
@pytest.mark.req("SP-SF-049")
def test_core_wheel_metadata_and_file_inventory(built_wheel: Path) -> None:
    """The pure Python core wheel contains only core files and the core runtime dependency."""
    metadata, files = _wheel_metadata(built_wheel)

    assert metadata["Name"] == "speechflow"
    assert frozenset(metadata["Requires-Python"].split(",")) == {">=3.12", "<3.14"}
    assert {_requirement_signature(requirement) for requirement in metadata.get_all("Requires-Dist", [])} == {
        _requirement_signature("numpy>=1.26.4")
    }
    assert metadata.get_all("Provides-Extra", []) == []
    assert built_wheel.name.endswith("-py3-none-any.whl")

    package_files = {path for path in files if ".dist-info/" not in path}
    assert all(path.startswith("speechflow/") for path in package_files)
    assert "speechflow/__init__.py" in package_files
    assert "speechflow/py.typed" in package_files
    assert not any(path.startswith("speechflow/audio/") for path in package_files)
    assert "speechflow/core/pcm.py" not in package_files
    assert not any(path.endswith((".so", ".dll", ".dylib", ".pyd")) for path in package_files)
    assert all(
        "/" not in path.removeprefix("speechflow/") or path.startswith("speechflow/core/") for path in package_files
    )


@pytest.mark.req("SP-SF-053")
@pytest.mark.req("SP-SF-054")
@pytest.mark.req("SP-SF-059")
@pytest.mark.req("SP-SF-060")
def test_core_public_and_module_author_surfaces_are_available() -> None:
    """Core consumer and module author imports are available in an isolated process."""
    script = f"""
import speechflow
from speechflow import core
from speechflow.core import (
    AudioData, ConfigurationError, EmptyOutputError, EngineCapabilities, EngineClosedError,
    EngineState, EngineStateError, EnvironmentUnavailableError, HealthSnapshot, InferenceError,
    ModuleNotInstalledError, TTSEngineBase, TTSError, iterate_async, iterate_sync, run_sync,
)

assert set(speechflow.__all__) == {CORE_EXPORTS!r}
for name in {CORE_EXPORTS - {"__version__"}!r}:
    assert globals()[name] is getattr(speechflow, name) is getattr(core, name)
for operation in (iterate_sync, run_sync, iterate_async, TTSEngineBase.prepare, TTSEngineBase.health):
    assert callable(operation)
"""
    result = _run_isolated(script)
    assert result.returncode == 0, result.stderr
