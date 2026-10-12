# Synchronous helper API

These helpers are available directly from `speechflow` and from `speechflow.core`.
Engines expose asynchronous operations. Synchronous callers use `run_sync()` to
prepare an engine, collect audio, and close the engine, and `iterate_sync()` to
consume its audio stream. See the [guide's synchronous example](../index.md#calling-from-synchronous-code).

`run_sync()` runs a coroutine on one shared daemon background event loop and waits
for the result. `iterate_sync()` reads an async iterator on that same loop. Keep
every operation on an engine on one event loop, including preparation and
closure. Use a separate engine instance for operations on an application's own
event loop.

Both helpers accept calls from other threads, including a caller with a running
event loop. Asynchronous code normally uses `await` and `async for`. Calling either
helper from its own background loop raises `EngineStateError`. Both helpers wait
for cleanup if the caller stops. Explicitly close a partially consumed synchronous
iterator, for example with `contextlib.closing`, and call
`run_sync(engine.aclose())` in a `finally` block.

`iterate_async(iterable)` reads a synchronous iterable in other threads without
blocking the caller's event loop. Early close or cancellation waits for an active
read and closes a closeable source. Iterator errors propagate unchanged.

```python
import asyncio

from speechflow import iterate_async


async def main() -> None:
    async for value in iterate_async([1, 2, 3]):
        print(value)


asyncio.run(main())
```

::: speechflow.core.sync
    options:
      members: [run_sync, iterate_sync, iterate_async]
