from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

import speechflow


@pytest.mark.req("SP-SF-003")
def test_registered_module_resolves_when_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An installed module added to the known module table resolves through its package attribute."""
    name = "published_engine"
    module_name = f"speechflow.{name}"
    (tmp_path / f"{name}.py").write_text("engine_name = 'published engine'\n", encoding="utf-8")
    distributions = cast(dict[str, str], vars(speechflow)["_MODULE_DISTRIBUTIONS"])
    monkeypatch.setitem(distributions, name, "speechflow-published-engine")
    monkeypatch.setattr(speechflow, "__path__", [*speechflow.__path__, str(tmp_path)])

    try:
        module = getattr(speechflow, name)
        assert isinstance(module, ModuleType)
        assert module.__name__ == module_name
        assert module.engine_name == "published engine"
        assert getattr(speechflow, name) is module
    finally:
        vars(speechflow).pop(name, None)
        sys.modules.pop(module_name, None)


@pytest.mark.req("SP-SF-060")
def test_missing_registered_module_has_distribution_specific_install_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent module added to the table raises a TTS and import error with distribution specific guidance."""
    name = "missing_published_engine"
    distribution = "speechflow-missing-published-engine"
    distributions = cast(dict[str, str], vars(speechflow)["_MODULE_DISTRIBUTIONS"])
    monkeypatch.setitem(distributions, name, distribution)

    with pytest.raises(speechflow.ModuleNotInstalledError) as caught:
        getattr(speechflow, name)

    assert isinstance(caught.value, speechflow.TTSError)
    assert isinstance(caught.value, ImportError)
    assert f"why=speechflow.{name}" in str(caught.value)
    assert f"what={distribution}" in str(caught.value)
    assert f"how=pip install {distribution}" in str(caught.value)
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == f"speechflow.{name}"
