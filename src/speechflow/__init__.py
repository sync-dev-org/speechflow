"""SpeechFlow's engine-neutral async TTS contracts."""

from __future__ import annotations

import importlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .core import (
    AudioData,
    ConfigurationError,
    EmptyOutputError,
    EngineCapabilities,
    EngineClosedError,
    EngineState,
    EngineStateError,
    EnvironmentUnavailableError,
    HealthSnapshot,
    InferenceError,
    ModuleNotInstalledError,
    TTSEngineBase,
    TTSError,
    iterate_async,
    iterate_sync,
    run_sync,
)

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "AudioData",
    "ConfigurationError",
    "EmptyOutputError",
    "EngineCapabilities",
    "EngineClosedError",
    "EngineState",
    "EngineStateError",
    "EnvironmentUnavailableError",
    "HealthSnapshot",
    "InferenceError",
    "ModuleNotInstalledError",
    "TTSEngineBase",
    "TTSError",
    "__version__",
    "iterate_async",
    "iterate_sync",
    "run_sync",
]

_MODULE_DISTRIBUTIONS: dict[str, str] = {}


def __getattr__(name: str) -> object:
    try:
        distribution = _MODULE_DISTRIBUTIONS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotInstalledError(
            f"why={module_name} is unavailable because {distribution} is not installed; "
            f"what={distribution} provides {module_name}; "
            f"how=pip install {distribution}"
        ) from exc
    globals()[name] = module
    return module
