"""Core data and shared asynchronous synthesis contract."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .exceptions import (
    ConfigurationError,
    EmptyOutputError,
    EngineClosedError,
    EngineStateError,
    InferenceError,
    TTSError,
)
from .sync import _wait_finished


@dataclass
class AudioData:
    """Lightweight container for PCM audio samples with metadata.

    The `data` array holds float32 samples normalized to the range [-1, 1]
    by convention. This range is enforced by engines that produce AudioData,
    not by this class itself. AudioData performs no validation at construction
    time — it is a plain data container.

    Note: the `format` field name shadows the Python builtin `format`. This is
    an accepted trade-off to keep the attribute name descriptive and consistent
    with the broader codebase convention.
    """

    data: np.ndarray  # float32, shape [samples] or [samples, channels]
    sample_rate: int
    channels: int = 1
    format: str = "pcm"


@dataclass(frozen=True)
class EngineCapabilities:
    """Static capabilities declared by an engine module.

    ``sample_rate`` is ``None`` when the output rate is unknown until the model
    is loaded; each returned ``AudioData`` then carries the actual rate.
    """

    languages: frozenset[str]
    auto_language: bool
    streaming: bool
    voice_listing: bool
    voice_cloning: bool
    sample_rate: int | None


class EngineState(StrEnum):
    """Preparation and lifetime state of an engine."""

    UNPREPARED = "unprepared"
    READY = "ready"
    CLOSED = "closed"


@dataclass(frozen=True)
class HealthSnapshot:
    """An immediate snapshot of the engine's local state."""

    state: EngineState
    busy: bool
    last_progress_monotonic_s: float | None


class TTSEngineBase(ABC):
    """Own preparation, FIFO synthesis, cancellation, and resource lifetime.

    Use one instance on one event loop. Modules implement ``_prepare`` (load
    and trial inference), ``_synthesize`` (incremental native synthesis), and
    ``_release`` (return resources), plus their static ``capabilities`` and
    immediate voice-list reads. A module whose capabilities declare
    ``voice_listing`` provides a synchronous ``list_voices()`` returning
    ``list[dict[str, str]]``, one ``{"name": ..., "kind": ...}`` per voice.
    Native hooks must not finish cancellation
    cleanup until their underlying work has actually stopped. Failed loading
    cleans its own partial resources; only successful loading is released here.
    """

    def __init__(self) -> None:
        self._prepared = False
        self._closed = False
        self._model_lock = asyncio.Lock()
        self._operations: set[asyncio.Task[Any]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._last_progress: float | None = None

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return the module's immutable static declaration without doing work."""
        ...

    @abstractmethod
    async def _prepare(self) -> None:
        """Load resources and run trial inference, cleaning failed partial loads.

        If cancellation cannot stop loading, wait for its real end. Returning
        normally records successful loading even when the caller was cancelled.
        """
        ...

    @abstractmethod
    def _synthesize(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncIterator[AudioData]:
        """Yield native output; finish cleanup only after native inference ends.

        A hook may catch cancellation, wait for inference to finish, and yield
        its final chunk. The parent discards that chunk and closes the source
        without requesting more output, waiting for the hook's cleanup.
        """
        ...

    @abstractmethod
    async def _release(self) -> None:
        """Return successfully prepared resources, after native work has ended."""
        ...

    def health(self) -> HealthSnapshot:
        """Read local state immediately, including while closing or synthesizing."""
        state = EngineState.CLOSED if self._closed else EngineState.READY if self._prepared else EngineState.UNPREPARED
        return HealthSnapshot(state, self._model_lock.locked(), self._last_progress)

    def _require_open(self) -> None:
        if self._closed:
            raise EngineClosedError("This engine is closed; create a new engine instance")

    def _require_ready(self) -> None:
        self._require_open()
        if not self._prepared:
            raise EngineStateError("Synthesis requires a prepared engine; await prepare() first")

    def _validate_synthesis(self, text: str) -> None:
        self._require_open()
        if not isinstance(text, str) or not text.strip():
            raise ConfigurationError("text must contain non-whitespace characters")

    def _start[T](self, work: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        task = asyncio.create_task(self._execute(work))
        self._operations.add(task)

        def finished(done: asyncio.Task[T]) -> None:
            self._operations.discard(done)
            # A queued task can be cancelled before its first instruction.
            if inspect.getcoroutinestate(work) == inspect.CORO_CREATED:
                work.close()

        task.add_done_callback(finished)
        return task

    async def _execute[T](self, work: Coroutine[Any, Any, T]) -> T:
        try:
            async with self._model_lock:
                self._require_open()
                result = await work
                self._require_open()
                return result
        except asyncio.CancelledError:
            if self._closed:
                raise EngineClosedError("Closing this engine cancelled the model operation") from None
            raise
        except TTSError:
            raise
        except Exception as exc:
            raise InferenceError(f"{type(self).__name__} model operation failed: {exc}") from exc
        finally:
            if inspect.getcoroutinestate(work) == inspect.CORO_CREATED:
                work.close()

    @staticmethod
    def _cancel(task: asyncio.Task[Any]) -> None:
        # Sending cancellation again could interrupt a module's native cleanup.
        if not task.done() and not task.cancelling():
            task.cancel()

    @staticmethod
    def _raise_if_cancelled() -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError

    async def _await_operation[T](self, task: asyncio.Task[T]) -> T:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            caller_cancelled = caller is not None and bool(caller.cancelling())
            self._cancel(task)
            await _wait_finished(task)
            if not caller_cancelled and self._closed:
                raise EngineClosedError("Closing this engine cancelled the model operation") from None
            raise

    async def prepare(self) -> None:
        """Load and trial the model once; callers own any timeout.

        A prepared engine returns immediately without waiting for synthesis.
        Concurrent loading requests queue behind the current load. Failures
        leave the engine unprepared and a later request retries loading.
        """
        self._require_open()
        if self._prepared:
            return
        await self._await_operation(self._start(self._prepare_once()))

    async def _prepare_once(self) -> None:
        if not self._prepared:
            await self._prepare()
            self._prepared = True

    async def get(self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object) -> AudioData:
        """Queue synthesis and return all samples after successful completion."""
        self._validate_synthesis(text)
        return await self._await_operation(self._start(self._collect(text, model, voice, kwargs)))

    async def _collect(self, text: str, model: str | None, voice: str | None, kwargs: dict[str, object]) -> AudioData:
        self._require_ready()
        source = self._synthesize(text, model=model, voice=voice, **kwargs)
        chunks: list[AudioData] = []
        try:
            async for chunk in source:
                self._raise_if_cancelled()
                chunks.append(chunk)
                self._last_progress = time.monotonic()
            self._raise_if_cancelled()
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()
        if not sum(chunk.data.shape[0] for chunk in chunks):
            raise EmptyOutputError("Synthesis completed without producing audio samples")
        first = chunks[0]
        audio = AudioData(
            np.concatenate([chunk.data for chunk in chunks]), first.sample_rate, first.channels, first.format
        )
        self._last_progress = time.monotonic()
        return audio

    async def stream(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncGenerator[AudioData, None]:
        """Yield output incrementally, holding the model until exhausted or closed.

        Close a partially read iterator explicitly (for example with
        ``contextlib.aclosing``) so native cleanup finishes before reuse.
        """
        self._validate_synthesis(text)
        requests: asyncio.Queue[asyncio.Future[AudioData]] = asyncio.Queue()
        task = self._start(self._serve_stream(text, model, voice, kwargs, requests))
        try:
            while True:
                reply = asyncio.get_running_loop().create_future()
                requests.put_nowait(reply)
                try:
                    done, _ = await asyncio.wait((reply, task), return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    self._cancel(task)
                    await _wait_finished(task)
                    raise
                if task in done:
                    await self._await_operation(task)
                    return
                yield reply.result()
        except GeneratorExit:
            self._cancel(task)
            await _wait_finished(task, propagate_cancel=True)
            if not task.cancelled():
                with contextlib.suppress(EngineClosedError):
                    task.result()
            raise
        finally:
            self._cancel(task)
            await _wait_finished(task)

    async def _serve_stream(
        self,
        text: str,
        model: str | None,
        voice: str | None,
        kwargs: dict[str, object],
        requests: asyncio.Queue[asyncio.Future[AudioData]],
    ) -> None:
        self._require_ready()
        source = self._synthesize(text, model=model, voice=voice, **kwargs)
        samples = 0
        try:
            while True:
                reply = await requests.get()
                try:
                    chunk = await anext(source)
                except StopAsyncIteration:
                    self._raise_if_cancelled()
                    break
                self._raise_if_cancelled()
                samples += chunk.data.shape[0]
                self._last_progress = time.monotonic()
                reply.set_result(chunk)
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()
        if not samples:
            raise EmptyOutputError("Synthesis completed without producing audio samples")
        self._last_progress = time.monotonic()

    async def aclose(self) -> None:
        """Cancel pending model work, wait for its real end, and release once."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        elif self._close_task.done():
            return
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _wait_finished(self._close_task)
            raise

    async def _finish_close(self) -> None:
        pending = tuple(self._operations)
        for task in pending:
            self._cancel(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        try:
            if self._prepared:
                await self._release()
        except TTSError:
            raise
        except Exception as exc:
            raise InferenceError(f"{type(self).__name__} resource release failed: {exc}") from exc
        finally:
            self._prepared = False

    async def __aenter__(self) -> TTSEngineBase:
        await self.prepare()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
