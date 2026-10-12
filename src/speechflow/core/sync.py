"""Generic bridges to one shared background event loop and thread-based reads."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import inspect
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Iterable, Iterator
from concurrent.futures import Future
from typing import Any, cast

from .exceptions import EngineStateError

_SIGNAL_CHECK_INTERVAL = 0.1


async def _wait_finished(task: asyncio.Task[Any], *, propagate_cancel: bool = False) -> None:
    """Wait through repeated caller cancellation without cancelling native work."""
    caller = asyncio.current_task()
    initial_cancelling = caller.cancelling() if caller is not None else 0
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if caller is not None and caller.cancelling() > initial_cancelling:
                cancellation = exc
        except Exception:
            pass
    if not task.cancelled():
        task.exception()  # Retrieve failures even when the caller remains cancelled.
    if propagate_cancel and cancellation is not None:
        raise cancellation


def _close_unstarted(coroutine: Coroutine[Any, Any, Any]) -> None:
    if inspect.iscoroutine(coroutine) and inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
        coroutine.close()


class _SyncRunner:
    """Run async work on one lazily started daemon event-loop thread."""

    def __init__(self) -> None:
        self._start_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def run[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Submit work, cancelling and waiting for cleanup if the caller stops."""
        try:
            self._raise_on_reentry("run_sync", "await")
            loop = self._ensure_started()
        except BaseException:
            _close_unstarted(coroutine)
            raise

        finished = threading.Event()
        interrupted = threading.Event()
        running: list[asyncio.Task[Any]] = []

        async def invoke() -> T:
            task = asyncio.current_task()
            assert task is not None
            running.append(task)
            try:
                if interrupted.is_set():
                    _close_unstarted(coroutine)
                    raise asyncio.CancelledError
                return await coroutine
            finally:
                finished.set()

        future: Future[T] = Future()

        def completed(task: asyncio.Task[T], destination: Future[T] = future) -> None:
            if task.cancelled():
                destination.cancel()
            elif (failure := task.exception()) is not None:
                destination.set_exception(failure)
            else:
                destination.set_result(task.result())

        def dispatch(destination: Future[T] = future, transfer: Callable[[asyncio.Task[T]], None] = completed) -> None:
            if interrupted.is_set():
                _close_unstarted(coroutine)
                destination.cancel()
                finished.set()
            else:
                task = loop.create_task(invoke())
                task.add_done_callback(transfer)

        def cancel() -> None:
            if running:
                task = running[0]
                if not task.done() and not task.cancelling():
                    task.cancel()
            else:
                # This also handles interruption before the dispatch callback.
                # A later invocation observes interrupted before doing any work.
                _close_unstarted(coroutine)
                finished.set()

        try:
            # Own the Future before scheduling: SIGINT may arrive after the
            # callback has been queued but before call_soon_threadsafe returns.
            loop.call_soon_threadsafe(dispatch)
            # A signal handled by another OS thread only marks Python's pending
            # handler; bounded waits let this caller return to Python to run it.
            while True:
                try:
                    return future.result(timeout=_SIGNAL_CHECK_INTERVAL)
                except TimeoutError:
                    if future.done():
                        # Preserve TimeoutError raised by the operation itself,
                        # or a result that arrived as the timed wait expired.
                        return future.result()
        except BaseException:
            interrupted.set()
            if not finished.is_set():
                if loop.is_closed():
                    _close_unstarted(coroutine)
                else:
                    loop.call_soon_threadsafe(cancel)
                    while not finished.is_set():
                        with contextlib.suppress(BaseException):
                            finished.wait(timeout=_SIGNAL_CHECK_INTERVAL)
            raise
        finally:
            # A concurrent Future's traceback can otherwise retain this frame.
            del future, completed, dispatch

    def iterate[T](self, async_iterator: AsyncIterator[T]) -> Iterator[T]:
        """Reject recursive use now, then yield one async item at a time."""
        self._raise_on_reentry("iterate_sync", "async for")

        def consume() -> Iterator[T]:
            try:
                while True:
                    try:
                        yield self.run(cast(Coroutine[Any, Any, T], anext(async_iterator)))
                    except StopAsyncIteration:
                        return
            finally:
                close = getattr(async_iterator, "aclose", None)
                if close is not None:
                    self.run(cast(Coroutine[Any, Any, None], close()))

        return consume()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        thread = self._thread
        if loop is not None and thread is not None and thread.is_alive():
            return loop
        with self._start_lock:
            loop = self._loop
            thread = self._thread
            if loop is not None and thread is not None and thread.is_alive():
                return loop
            ready = threading.Event()
            thread = threading.Thread(
                target=self._run_loop,
                args=(ready,),
                name="speechflow-sync-runner",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            ready.wait()
            loop = self._loop
            if loop is None:
                raise EngineStateError("SpeechFlow background event loop failed to start")
            return loop

    def _raise_on_reentry(self, function: str, alternative: str) -> None:
        if threading.current_thread() is self._thread:
            raise EngineStateError(
                f"{function} cannot be called from the SpeechFlow background event loop thread; "
                f"use {alternative} instead"
            )

    def _run_loop(self, ready: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _cancel_pending(self) -> None:
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            if not task.cancelling():
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.get_running_loop().shutdown_asyncgens()

    def shutdown(self) -> None:
        """Stop the daemon loop on process exit when it is still reachable."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None or not thread.is_alive():
            return
        if threading.current_thread() is thread:
            loop.stop()
            return
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._cancel_pending(), loop).result(timeout=1)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1)


_RUNNER = _SyncRunner()
atexit.register(_RUNNER.shutdown)


def run_sync[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on the shared loop and wait, including cancellation cleanup.

    Call from any thread except the background loop's own thread. An async
    caller should use ``await`` there instead.
    """
    return _RUNNER.run(coroutine)


def iterate_sync[T](async_iterator: AsyncIterator[T]) -> Iterator[T]:
    """Read async items synchronously, closing the source on early termination.

    Background-loop callers are rejected immediately; use ``async for`` there.
    Explicitly close a partially consumed iterator to wait for its cleanup.
    """
    return _RUNNER.iterate(async_iterator)


async def _read_thread[T](function: Callable[..., T], *args: object) -> T:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling a to_thread task does not stop its OS thread. Wait for the
        # read before closing the iterator or returning to the caller.
        await _wait_finished(task)
        raise


def _next[T](iterator: Iterator[T]) -> tuple[bool, T | None]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def iterate_async[T](iterable: Iterable[T]) -> AsyncGenerator[T, None]:
    """Read a synchronous iterable in other threads without blocking the loop.

    Early close and cancellation wait for any running read, then close a
    closeable source in another thread. Iterator errors propagate unchanged.
    """
    creation = asyncio.create_task(asyncio.to_thread(iter, iterable))
    iterator: Iterator[T] | None = None
    try:
        try:
            iterator = await asyncio.shield(creation)
        except asyncio.CancelledError:
            await _wait_finished(creation)
            raise
        while True:
            available, value = await _read_thread(_next, iterator)
            if not available:
                return
            yield cast(T, value)
    finally:
        if iterator is None and creation.done() and not creation.cancelled() and creation.exception() is None:
            iterator = creation.result()
        close = getattr(iterator, "close", None)
        if close is not None:
            await _read_thread(close)
