"""Failures can be caught using domain and Python exception families."""

import pytest

import speechflow


@pytest.mark.req("SP-SF-060")
@pytest.mark.parametrize(
    ("name", "standard", "domain_parent"),
    [
        ("ConfigurationError", ValueError, "TTSError"),
        ("EngineStateError", RuntimeError, "TTSError"),
        ("EngineClosedError", RuntimeError, "EngineStateError"),
        ("InferenceError", RuntimeError, "TTSError"),
        ("EmptyOutputError", RuntimeError, "InferenceError"),
        ("EnvironmentUnavailableError", RuntimeError, "TTSError"),
        ("ModuleNotInstalledError", ImportError, "TTSError"),
    ],
)
def test_error_has_domain_and_standard_exception_families(
    name: str, standard: type[Exception], domain_parent: str
) -> None:
    """error の種類は約束した意味の Python 標準の error と speechflow の error を兼ねる。"""
    failure = getattr(speechflow, name)("failure")
    assert isinstance(failure, standard)
    assert isinstance(failure, getattr(speechflow, domain_parent))
    assert isinstance(failure, speechflow.TTSError)


@pytest.mark.req("SP-SF-053")
@pytest.mark.req("SP-SF-054")
@pytest.mark.req("SP-SF-059")
@pytest.mark.req("SP-SF-060")
def test_root_and_core_export_the_new_contract() -> None:
    """package root と core は同じ engine の型、error、同期の関数を公開する。"""
    expected = {
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
    }
    assert set(speechflow.__all__) == expected | {"__version__"}
    assert set(speechflow.core.__all__) == expected
    for name in expected:
        assert getattr(speechflow, name) is getattr(speechflow.core, name)
