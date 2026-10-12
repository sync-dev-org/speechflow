# Error API

These errors are available directly from `speechflow` and from `speechflow.core`.
Catch `TTSError` for engine failures or a subclass for specific recovery.

| Error | Meaning | Parent classes |
| --- | --- | --- |
| `TTSError` | Base engine error | `Exception` |
| `ConfigurationError` | Invalid input or configuration | `TTSError`, `ValueError` |
| `EngineStateError` | Operation incompatible with the engine's state | `TTSError`, `RuntimeError` |
| `EngineClosedError` | Closed engine or operation cancelled by closing | `EngineStateError` |
| `InferenceError` | Model operation or resource release failure | `TTSError`, `RuntimeError` |
| `EmptyOutputError` | Nonempty text produced no audio samples | `InferenceError` |
| `EnvironmentUnavailableError` | Preparation cannot run in the available environment | `TTSError`, `RuntimeError` |
| `ModuleNotInstalledError` | A registered engine module is not installed | `TTSError`, `ImportError` |

Invalid or whitespace-only text raises `ConfigurationError`. Synthesis before
preparation raises `EngineStateError`, and use after closing raises
`EngineClosedError`. A completed response or stream with no audio samples raises
`EmptyOutputError`. Model operation and resource release failures are translated
to `InferenceError` with the original exception as their cause. Engine errors
already in the `TTSError` family propagate unchanged. Environment problems are
reported during preparation.

Caller cancellation remains `asyncio.CancelledError`; closing the engine reports
cancelled operations with `EngineClosedError`. See the [engine lifecycle](engine.md).

The core's registered module table is currently empty. Import an engine module
explicitly from its module root. Python raises `ModuleNotFoundError` for an
uninstalled explicit import. Only registered package attributes use
`ModuleNotInstalledError` when their module is missing; unknown attributes raise
`AttributeError`.

::: speechflow.core.exceptions
    options:
      members: null
      filters: ["!^_"]
