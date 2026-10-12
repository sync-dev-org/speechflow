# Engines and health API

These names are available directly from `speechflow` and from `speechflow.core`.
The core owns preparation, operation ordering, cancellation, closure, and common
health. Engine modules implement model loading, synthesis, and resource release.
Use one engine instance on one event loop throughout its lifetime.

## Engine contract

`prepare()` loads and trials the model once after success. A failed preparation
leaves the engine unprepared and allows a later call to retry. The caller chooses
the timeout. `async with` prepares the engine on entry and closes it on exit.

`get(text, model=None, voice=None)` collects the complete output into one
`AudioData`. `stream(text, model=None, voice=None)` yields `AudioData` chunks.
Engine-specific options are additional keyword arguments. Model operations on
one engine run one at a time in arrival order. Synthesis requires preparation.
Close a partially consumed stream explicitly, for example with
`contextlib.aclosing`, so cleanup finishes before reusing the engine.

`aclose()` cancels pending model operations, waits for work and cleanup to finish,
and releases resources. Repeated closing is safe. A closed instance cannot be
reused. Caller cancellation also waits for the underlying work to finish; the
core does not promise how quickly it stops. Engine closure reports cancelled
operations with `EngineClosedError`; caller cancellation remains
`asyncio.CancelledError`. See the [error API](errors.md).

Subclass constructors call `super().__init__()`. Implement `_prepare()` to load
resources and run trial inference, `_synthesize()` to yield audio, and
`_release()` to return resources. Failed loading cleans up its own partial
resources. Native hooks wait for underlying work to stop before finishing
cancellation cleanup. Each engine module is responsible for producing audio
that meets the [AudioData contract](audio.md).

::: speechflow.core.base.TTSEngineBase
    options:
      members: [capabilities, prepare, get, stream, aclose, health, _prepare, _synthesize, _release]
      filters: ["!^__"]

## Capabilities and voices

`EngineCapabilities` is an immutable declaration with `languages`,
`auto_language`, `streaming`, `voice_listing`, `voice_cloning`, and `sample_rate`.
When `sample_rate` is `None`, the output rate is unknown until the model is loaded.
Use the actual rate carried by each returned `AudioData`.

A module with `voice_listing=True` provides a synchronous `list_voices()` returning
`list[dict[str, str]]`. Each element has the shape
`{"name": <voice name>, "kind": <voice kind>}`. A module with
`voice_listing=False` does not provide this operation. The core base class does
not define it.

::: speechflow.core.base.EngineCapabilities

## Health

Read `engine.capabilities` and `engine.health()` synchronously without waiting for
model work. `HealthSnapshot` carries `state` (`EngineState.UNPREPARED`,
`EngineState.READY`, or `EngineState.CLOSED`), `busy`, and
`last_progress_monotonic_s`. The state values are `"unprepared"`, `"ready"`, and
`"closed"`. These reads remain available after closing.

::: speechflow.core.base.EngineState

::: speechflow.core.base.HealthSnapshot
