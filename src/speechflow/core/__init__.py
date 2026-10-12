"""Engine-neutral synthesis types, operations, and failures."""

from .base import AudioData, EngineCapabilities, EngineState, HealthSnapshot, TTSEngineBase
from .exceptions import (
    ConfigurationError,
    EmptyOutputError,
    EngineClosedError,
    EngineStateError,
    EnvironmentUnavailableError,
    InferenceError,
    ModuleNotInstalledError,
    TTSError,
)
from .sync import iterate_async, iterate_sync, run_sync

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
    "iterate_async",
    "iterate_sync",
    "run_sync",
]
