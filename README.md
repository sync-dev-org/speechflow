# SpeechFlow

English | [日本語](https://github.com/sync-dev-org/speechflow/blob/main/README.ja.md)

SpeechFlow is an asynchronous Python library for text-to-speech (TTS). Its core
provides shared engine contracts, audio data, errors, and functions for calling
asynchronous operations from synchronous code. Engine modules are separate
distributions; the core depends only on NumPy and has no optional extras.

[Source repository](https://github.com/sync-dev-org/speechflow)

## Installation

SpeechFlow supports Python 3.12 and Python 3.13. Install the core from PyPI:

```bash
pip install speechflow
```

Or add it to a uv project:

```bash
uv add speechflow
```

The core does not include a voice model or a synthesis engine. Each engine module
documents its own installation, supported platforms, and model requirements.

## Quick Start

This example runs with only the core and its NumPy dependency. It emits two chunks
of silence to demonstrate the engine API; it does not convert text into speech.
A real module implements model loading and trial inference in `_prepare()`, yields
its audio in `_synthesize()`, and returns its resources in `_release()`.

```python
import asyncio
from collections.abc import AsyncIterator

import numpy as np

from speechflow import AudioData, EngineCapabilities, TTSEngineBase


class ExampleTTSEngine(TTSEngineBase):
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            languages=frozenset(),
            auto_language=False,
            streaming=True,
            voice_listing=False,
            voice_cloning=False,
            sample_rate=24000,
        )

    async def _prepare(self) -> None:
        pass

    async def _synthesize(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncIterator[AudioData]:
        for _ in range(2):
            yield AudioData(np.zeros(2400, dtype=np.float32), sample_rate=24000)

    async def _release(self) -> None:
        pass


async def main() -> None:
    async with ExampleTTSEngine() as engine:
        audio = await engine.get("Hello from SpeechFlow.")
        print(audio.sample_rate, audio.data.shape)
        async for chunk in engine.stream("Stream an example."):
            print(chunk.sample_rate, chunk.data.shape)


asyncio.run(main())
```

`async with` prepares the engine before entering the block and closes it on exit.
`get()` collects the complete output; `stream()` yields `AudioData` chunks.
The examples below reuse `ExampleTTSEngine` from this section.

## Engine Lifecycle

Prepare an engine before synthesis. Preparation loads and trials the model once;
a later call on a prepared engine returns immediately. If preparation fails, the
engine remains unprepared and a later call can retry. The caller chooses any
timeout, for example with `asyncio.timeout()`:

```python
async def synthesize_once() -> AudioData:
    engine = ExampleTTSEngine()
    try:
        async with asyncio.timeout(30):
            await engine.prepare()
        return await engine.get("Prepare explicitly.", model=None, voice=None)
    finally:
        await engine.aclose()


print(asyncio.run(synthesize_once()).data.shape)
```

Use one engine instance on one event loop throughout preparation, synthesis, and
closing. A closed instance cannot be reused; create a new instance instead.
`aclose()` cancels pending model operations, waits for their work and cleanup to
finish, and releases resources. Repeated closing is safe. Cancellation also waits
for the underlying work to finish; the core does not promise how quickly it stops.

Model operations on one engine run one at a time, in arrival order. Read
`engine.capabilities` and `engine.health()` synchronously without waiting for model
work. A health snapshot carries `state` (`unprepared`, `ready`, or `closed`),
`busy`, and `last_progress_monotonic_s`.

### Streaming Cleanup

Close a partially consumed stream explicitly so its cleanup finishes before the
engine is reused. `contextlib.aclosing` handles early exit from an async loop:

```python
from contextlib import aclosing


async def first_chunk() -> None:
    async with ExampleTTSEngine() as engine:
        async with aclosing(engine.stream("Read one chunk.")) as chunks:
            async for chunk in chunks:
                print(chunk.data.shape)
                break


asyncio.run(first_chunk())
```

### Calling from Synchronous Code

`speechflow.run_sync()` runs a coroutine on one shared daemon background event
loop and waits for the result. `speechflow.iterate_sync()` reads an async iterator
on that same loop. Both functions wait for cleanup if the caller stops. Explicitly
close a partially consumed synchronous iterator, for example with
`contextlib.closing`:

```python
from contextlib import closing

import speechflow

engine = ExampleTTSEngine()
try:
    speechflow.run_sync(engine.prepare())
    audio = speechflow.run_sync(engine.get("Hello from synchronous code."))
    print(audio.sample_rate, audio.data.shape)
    with closing(speechflow.iterate_sync(engine.stream("Read one chunk."))) as chunks:
        for chunk in chunks:
            print(chunk.data.shape)
            break
finally:
    speechflow.run_sync(engine.aclose())
```

Keep every operation on this engine on the shared background loop. Use a separate
engine instance for operations on an application's own loop. The bridge functions
accept calls from other threads, including a caller with a running event loop;
asynchronous code normally uses `await` and `async for`. Calling either bridge
from its own background loop raises `EngineStateError`.

`speechflow.iterate_async(iterable)` reads a synchronous iterable in other threads
without blocking the caller's event loop. Early close or cancellation waits for an
active read and closes a closeable source.

## Public API

The package root and `speechflow.core` export these names:

| Name | Purpose |
| --- | --- |
| `AudioData` | PCM samples and audio metadata |
| `TTSEngineBase` | Shared preparation, synthesis, cancellation, and closing behavior |
| `EngineCapabilities` | Immutable declaration of languages, streaming, voices, and output rate |
| `EngineState` | `UNPREPARED`, `READY`, and `CLOSED` states |
| `HealthSnapshot` | Immediate state, busy flag, and last progress time |
| `run_sync`, `iterate_sync`, `iterate_async` | Bridges between asynchronous and synchronous code |
| `TTSError` | Base engine error |
| `ConfigurationError` | Invalid input or configuration |
| `EngineStateError` | Operation incompatible with the engine's state |
| `EngineClosedError` | Closed engine or operation cancelled by closing |
| `InferenceError` | Model operation or resource release failure |
| `EmptyOutputError` | Nonempty text produced no audio samples |
| `EnvironmentUnavailableError` | Preparation cannot run in the available environment |
| `ModuleNotInstalledError` | A registered engine module is not installed |

`speechflow.__version__` reports the core distribution version. Engine classes are
exported from their module roots. Their `get(text, model=None, voice=None)` and
`stream(text, model=None, voice=None)` operations preserve the base arguments;
module-specific options are additional keyword arguments.

### Capabilities and Voices

`EngineCapabilities` carries `languages`, `auto_language`, `streaming`,
`voice_listing`, `voice_cloning`, and `sample_rate`. When `sample_rate` is `None`,
the output rate is unknown until the model is loaded. Each returned `AudioData`
carries the actual rate; use that value rather than assuming a fixed rate.

A module with `voice_listing=True` provides a synchronous `list_voices()` returning
`list[dict[str, str]]`. Each element has the shape
`{"name": <voice name>, "kind": <voice kind>}`. A module with
`voice_listing=False` does not provide this operation. The core base class does
not define it.

## AudioData

```python
import numpy as np

from speechflow import AudioData

audio = AudioData(
    data=np.array([0.0, 0.25, -0.25], dtype=np.float32),
    sample_rate=24000,
    channels=1,
    format="pcm",
)
```

Engine authors must return finite float32 samples normalized to `[-1, 1]`, normally
mono, with an explicit `sample_rate` in Hz. The `data` array must be a writable,
independent copy that does not share the engine's internal buffers. Mono data has
shape `[samples]`; multichannel data has shape `[samples, channels]`.
`channels` defaults to `1` and `format` to `"pcm"`.

`AudioData` is a plain container. It does not validate, normalize, or copy arrays
at construction time; producing output that meets these requirements is the
engine module's responsibility.

## Errors

Catch `TTSError` for engine failures or a subclass for specific recovery:

```python
from speechflow import ConfigurationError, EmptyOutputError, TTSError


async def handle_errors() -> None:
    async with ExampleTTSEngine() as engine:
        try:
            audio = await engine.get("Hello.")
            print(audio.data.shape)
        except ConfigurationError as exc:
            print(f"Configuration problem: {exc}")
        except EmptyOutputError as exc:
            print(f"Synthesis produced no audio: {exc}")
        except TTSError as exc:
            print(f"TTS failed: {exc}")


asyncio.run(handle_errors())
```

Invalid or whitespace-only text raises `ConfigurationError`. Synthesis before
preparation raises `EngineStateError`, and use after closing raises
`EngineClosedError`. A completed response or stream with no audio samples raises
`EmptyOutputError`. Model operation and resource release failures are translated
to `InferenceError` with the original exception as their cause; engine errors
already in the `TTSError` family propagate unchanged.

The core's registered module table is currently empty. Import an engine module
explicitly from its module root. Python raises `ModuleNotFoundError` for an
uninstalled explicit import. Only registered package attributes use
`ModuleNotInstalledError` when their module is missing; unknown attributes raise
`AttributeError`.

## Application Responsibilities

Applications own configuration, credentials, model weights and their acquisition,
external process management, and deployment. The core does not load `.env` files.
Audio file output, resampling, effects, echo cancellation, noise suppression,
speech recognition, and conversation management are outside the core's scope.

## Documentation

The [documentation guide](https://sync-dev-org.github.io/speechflow/) links to the
API reference for engines and lifecycle, synchronous helpers, audio data, and errors.

## Development

The [public repository](https://github.com/sync-dev-org/speechflow) contains the
core source and its tests. Verification requires no engine module checkout, GPU,
model weights, or native toolchain:

```bash
uv sync
uv lock --check
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy
uv build --wheel
uv run mkdocs build --strict
```

Each engine module repository owns its metadata, lock, source, tests, builds, and
distribution. The core repository does not collect or build those distributions.

## License

[MIT](https://github.com/sync-dev-org/speechflow/blob/main/LICENSE)
