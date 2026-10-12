"""Shared async interface declarations owned by the SpeechFlow core."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from speechflow import TTSEngineBase


def _assert_base_signature(method: Callable[..., object]) -> None:
    parameters = list(inspect.signature(method).parameters.values())
    assert [parameter.name for parameter in parameters[:4]] == ["self", "text", "model", "voice"]
    assert parameters[2].default is None
    assert parameters[3].default is None


@pytest.mark.req("SP-SF-003")
def test_base_get_declaration_is_async_with_the_unified_signature() -> None:
    """The base get operation is async and accepts text, model, and voice."""
    assert inspect.iscoroutinefunction(TTSEngineBase.get)
    _assert_base_signature(TTSEngineBase.get)


@pytest.mark.req("SP-SF-003")
def test_base_stream_declaration_is_not_a_coroutine_function() -> None:
    """The base stream accepts text, model, and voice without being a coroutine function."""
    assert not inspect.iscoroutinefunction(TTSEngineBase.stream)
    _assert_base_signature(TTSEngineBase.stream)
