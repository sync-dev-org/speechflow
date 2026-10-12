"""Generic bridges preserve asynchronous work and cleanup across calling threads."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import speechflow
from speechflow.core.sync import iterate_sync, run_sync


@pytest.mark.req("SP-SF-054")
@pytest.mark.parametrize("in_running_loop", [False, True])
def test_run_sync_shares_one_background_loop_across_threads(in_running_loop: bool) -> None:
    """同期の関数は複数の thread と実行中の event loop から一つの背景の loop を使う。"""

    async def identify(value: int) -> tuple[int, int, asyncio.AbstractEventLoop]:
        return value, threading.get_ident(), asyncio.get_running_loop()

    def call() -> list[tuple[int, int, asyncio.AbstractEventLoop]]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda value: run_sync(identify(value)), range(8)))

    async def inside() -> list[tuple[int, int, asyncio.AbstractEventLoop]]:
        return call()

    results = asyncio.run(inside()) if in_running_loop else call()
    assert [item[0] for item in results] == list(range(8))
    assert len({item[1] for item in results}) == 1
    assert results[0][1] != threading.get_ident()
    assert all(item[2] is results[0][2] for item in results)


@pytest.mark.req("SP-SF-054")
@pytest.mark.parametrize("in_running_loop", [False, True])
def test_iterate_sync_is_incremental_and_closes_the_source(in_running_loop: bool) -> None:
    """同期の列は音声を一つずつ返し、列を途中で閉じると非同期の後始末を待つ。"""
    cleaned = threading.Event()

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            raise AssertionError("The second item was never requested")
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    def consume() -> None:
        iterator = iterate_sync(source())
        assert next(iterator) == 1
        iterator.close()  # type: ignore[attr-defined]
        assert cleaned.is_set()

    async def inside() -> None:
        consume()

    if in_running_loop:
        asyncio.run(inside())
    else:
        consume()


@pytest.mark.req("SP-SF-054")
def test_iterate_sync_propagates_failure_and_closes_the_source() -> None:
    """同期の列は途中の失敗をそのまま知らせ、元の列を閉じる。"""
    closed = threading.Event()
    failure = LookupError("source failure")

    class Source:
        def __aiter__(self) -> Source:
            return self

        async def __anext__(self) -> int:
            raise failure

        async def aclose(self) -> None:
            closed.set()

    with pytest.raises(LookupError) as caught:
        next(iterate_sync(Source()))
    assert caught.value is failure
    assert closed.is_set()


@pytest.mark.req("SP-SF-054")
@pytest.mark.parametrize("coroutine_state", ["created", "suspended", "running"])
def test_run_sync_reentry_rejects_without_closing_started_coroutine(coroutine_state: str) -> None:
    """背景の loop から run_sync を呼ぶと、開始済みの coroutine の終了処理に上書きされず状態の誤りを知らせる。"""

    async def check() -> None:
        gate = asyncio.Event()

        async def dangerous() -> None:
            try:
                await gate.wait()
            finally:
                raise LookupError("Closing a started coroutine must not mask rejection")

        async def reject(coroutine: Coroutine[Any, Any, object]) -> None:
            with pytest.raises(speechflow.EngineStateError) as caught:
                run_sync(coroutine)
            message = str(caught.value)
            assert "run_sync" in message
            assert "background" in message and "thread" in message
            assert "await" in message

        if coroutine_state == "running":
            holder: list[Coroutine[Any, Any, object]] = []

            async def self_reentry() -> None:
                await reject(holder[0])

            holder.append(self_reentry())
            await holder[0]
        else:
            coroutine = dangerous()
            if coroutine_state == "suspended":
                coroutine.send(None)
            try:
                await reject(coroutine)
                if coroutine_state == "suspended":
                    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_SUSPENDED
            finally:
                with contextlib.suppress(LookupError):
                    coroutine.close()

    run_sync(check())


@pytest.mark.req("SP-SF-054")
def test_iterate_sync_reentry_rejects_when_called_before_requesting_an_item() -> None:
    """背景の loop から iterate_sync を呼ぶと item を求める前に関数名と async for を持つ error で断る。"""

    async def check() -> None:
        entered = False

        async def source() -> AsyncGenerator[int, None]:
            nonlocal entered
            entered = True
            yield 1

        stream = source()
        try:
            with pytest.raises(speechflow.EngineStateError) as caught:
                iterate_sync(stream)
            message = str(caught.value)
            assert "iterate_sync" in message
            assert "background" in message and "thread" in message
            assert "async for" in message
            assert not entered
        finally:
            await stream.aclose()

    run_sync(check())


@pytest.mark.req("SP-SF-054")
@pytest.mark.parametrize("operation", ["run_sync", "iterate_sync"])
def test_caller_interrupt_waits_for_cancelled_background_operation(operation: str) -> None:
    """同期の呼び出し元が中断されると背景の仕事を取り消し、その後始末を待ってから戻る。"""
    script = f"""
import asyncio
import os
import signal
import threading
from speechflow.core.sync import run_sync, iterate_sync
cleaned = threading.Event()
async def pending():
    try:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.Event().wait()
    finally:
        await asyncio.sleep(0)
        cleaned.set()
async def source():
    yield 1
    await pending()
    yield 2
try:
    if {operation!r} == "run_sync":
        run_sync(pending())
    else:
        iterator = iterate_sync(source())
        assert next(iterator) == 1
        next(iterator)
except KeyboardInterrupt:
    assert cleaned.is_set(), "returned before background cancellation cleanup"
else:
    raise AssertionError("caller interrupt was not propagated")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-SF-054")
@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="requires POSIX thread signal masks")
@pytest.mark.parametrize("operation", ["run_sync", "iterate_sync"])
@pytest.mark.parametrize("interrupt_during_cleanup", [False, True])
def test_interrupt_received_by_background_thread_cancels_work_and_waits_for_cleanup(
    operation: str, interrupt_during_cleanup: bool
) -> None:
    """背景の thread に届いた中断でも同期の呼び出しを取り消し、再度の中断を受けても後始末を待つ。"""
    script = f"""
import asyncio
import signal
import sys
import threading
from speechflow.core.sync import run_sync, iterate_sync

main_thread = threading.get_ident()
cleaned = threading.Event()
release_cleanup = None
background_loop = None
interrupts = 0

async def warm_up():
    global background_loop, release_cleanup
    background_loop = asyncio.get_running_loop()
    release_cleanup = asyncio.Event()

run_sync(warm_up())  # Start the background thread before masking the main thread.

def handle_interrupt(signum, frame):
    global interrupts
    interrupts += 1
    if interrupts == 1:
        raise KeyboardInterrupt
    background_loop.call_soon_threadsafe(release_cleanup.set)
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, handle_interrupt)
signal.pthread_sigmask(signal.SIG_BLOCK, {{signal.SIGINT}})

async def interrupt_waiting_caller():
    # Observe the real standard-library wait; no delay chooses the signal order.
    # The caller cannot run Python's handler while its native wait stays asleep.
    while sys._current_frames()[main_thread].f_code is not threading.Condition.wait.__code__:
        await asyncio.sleep(0)
    signal.pthread_kill(threading.get_ident(), signal.SIGINT)

async def pending():
    try:
        await interrupt_waiting_caller()
        await asyncio.Event().wait()
    finally:
        if {interrupt_during_cleanup!r}:
            print("cleanup waiting for second interrupt", flush=True)
            await interrupt_waiting_caller()
            await release_cleanup.wait()
        await asyncio.sleep(0)
        cleaned.set()

async def source():
    yield 1
    await pending()
    yield 2

try:
    if {operation!r} == "run_sync":
        run_sync(pending())
    else:
        iterator = iterate_sync(source())
        assert next(iterator) == 1
        next(iterator)
except KeyboardInterrupt:
    assert cleaned.is_set(), "returned before background cancellation cleanup"
    assert interrupts == (2 if {interrupt_during_cleanup!r} else 1)
else:
    raise AssertionError("caller interrupt was not propagated")
"""
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10, check=False)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{operation} did not return after interruption; stdout={exc.stdout!r}; stderr={exc.stderr!r}")
    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-SF-054")
@pytest.mark.parametrize("operation", ["run_sync", "iterate_sync"])
def test_sync_helpers_propagate_operation_timeout_errors_unchanged(operation: str) -> None:
    """同期の関数は背景の操作が出す時間切れの error をそのまま呼び出し側へ返す。"""
    failure = TimeoutError("operation timed out")

    async def fail() -> None:
        raise failure

    async def source() -> AsyncIterator[int]:
        await fail()
        yield 1

    with pytest.raises(TimeoutError) as caught:
        if operation == "run_sync":
            run_sync(fail())
        else:
            next(iterate_sync(source()))
    assert caught.value is failure


@pytest.mark.req("SP-SF-054")
async def test_iterate_async_reads_and_closes_sync_source_in_other_threads() -> None:
    """同期の列を読む非同期の列は、読み取りと後始末を呼び出し元とは別の thread で行う。"""
    caller = threading.get_ident()
    threads: list[int] = []
    closed = threading.Event()

    def source() -> Iterator[int]:
        try:
            for value in (1, 2):
                threads.append(threading.get_ident())
                yield value
        finally:
            threads.append(threading.get_ident())
            closed.set()

    assert [item async for item in speechflow.core.sync.iterate_async(source())] == [1, 2]
    assert closed.is_set()
    assert all(thread != caller for thread in threads)


@pytest.mark.req("SP-SF-054")
async def test_iterate_async_cancellation_waits_for_in_progress_read_then_closes() -> None:
    """同期の列の読み取り中の取り消しは読み取りの終了を待ち、元の列を閉じる。"""
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Source:
        def __iter__(self) -> Source:
            return self

        def __next__(self) -> int:
            started.set()
            assert release.wait(timeout=5)
            return 1

        def close(self) -> None:
            closed.set()

    stream = speechflow.core.sync.iterate_async(Source())
    task = asyncio.ensure_future(anext(stream))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert not closed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()


@pytest.mark.req("SP-SF-054")
async def test_iterate_async_early_close_and_failure_close_the_source() -> None:
    """非同期の列を途中で閉じた時も、同期の読み取りが失敗した時も元の列を閉じる。"""
    for fails in (False, True):
        closed = threading.Event()

        def source(fails: bool = fails, closed: threading.Event = closed) -> Iterator[int]:
            try:
                if fails:
                    raise OSError("read failure")
                yield 1
                yield 2
            finally:
                closed.set()

        stream = speechflow.core.sync.iterate_async(source())
        if fails:
            with pytest.raises(OSError, match="read failure"):
                await anext(stream)
        else:
            assert await anext(stream) == 1
            await stream.aclose()
        assert closed.is_set()


@pytest.mark.req("SP-SF-054")
async def test_iterate_async_cancellation_during_iterator_creation_closes_created_resource() -> None:
    """同期の列を作る途中の取り消しでも、作成が終わるのを待ち出来た列を閉じる。"""
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Source:
        def __iter__(self) -> Source:
            started.set()
            assert release.wait(timeout=5)
            return self

        def __next__(self) -> int:
            return 1

        def close(self) -> None:
            closed.set()

    stream = speechflow.core.sync.iterate_async(Source())
    task = asyncio.ensure_future(anext(stream))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
