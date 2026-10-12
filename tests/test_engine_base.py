"""Behavior of the engine contract shared by every synthesis module."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
from collections.abc import AsyncIterator, Coroutine
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import speechflow
from speechflow import AudioData, TTSEngineBase


def error(name: str) -> type[Exception]:
    return getattr(speechflow, name)  # type: ignore[no-any-return]


async def checkpoint() -> None:
    """Let already scheduled tasks enter their next await without a timed sleep."""
    ready = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(ready.set_result, None)
    await ready


async def submitted[T](work: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    entered = asyncio.Event()

    async def call() -> T:
        entered.set()
        return await work

    task = asyncio.create_task(call())
    await entered.wait()
    await checkpoint()
    return task


class SampleTTSEngine(TTSEngineBase):
    """Real subclass with observable, replaceable native work."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.prepare_started = asyncio.Event()
        self.prepare_gate = asyncio.Event()
        self.prepare_gate.set()
        self.prepare_cancelled = asyncio.Event()
        self.prepare_cleanup = asyncio.Event()
        self.prepare_cleanup.set()
        self.prepare_failure: Exception | None = None
        self.finish_prepare_on_cancel = False
        self.synthesis_started = asyncio.Event()
        self.synthesis_gate = asyncio.Event()
        self.synthesis_gate.set()
        self.synthesis_cancelled = asyncio.Event()
        self.synthesis_cleanup = asyncio.Event()
        self.synthesis_cleanup.set()
        self.release_started = asyncio.Event()
        self.release_gate = asyncio.Event()
        self.release_gate.set()
        self.release_failure: Exception | None = None
        self.synthesis_failure: Exception | None = None
        self.chunks = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.3], dtype=np.float32)]
        self.arguments: tuple[str, str | None, str | None, dict[str, object]] | None = None

    @property
    def capabilities(self) -> Any:
        return speechflow.EngineCapabilities(
            languages=frozenset({"en", "ja"}),
            auto_language=False,
            streaming=True,
            voice_listing=True,
            voice_cloning=False,
            sample_rate=24000,
        )

    def list_voices(self) -> list[str]:
        """Return a voice list held in memory, with no model access."""
        return ["default"]

    async def _prepare(self) -> None:
        self.calls.append("prepare")
        self.prepare_started.set()
        try:
            await self.prepare_gate.wait()
        except asyncio.CancelledError:
            self.prepare_cancelled.set()
            await self.prepare_cleanup.wait()
            if not self.finish_prepare_on_cancel:
                raise
        if self.prepare_failure is not None:
            raise self.prepare_failure
        self.calls.append("prepared")

    async def _synthesize(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncIterator[AudioData]:
        self.arguments = text, model, voice, kwargs
        self.calls.append(text)
        self.synthesis_started.set()
        try:
            for data in self.chunks:
                yield AudioData(data, sample_rate=24000)
                await self.synthesis_gate.wait()
            if self.synthesis_failure is not None:
                raise self.synthesis_failure
        finally:
            self.synthesis_cancelled.set()
            await self.synthesis_cleanup.wait()
            self.calls.append(f"finished:{text}")

    async def _release(self) -> None:
        self.calls.append("release")
        self.release_started.set()
        await self.release_gate.wait()
        if self.release_failure is not None:
            raise self.release_failure
        self.calls.append("released")


@pytest.mark.req("SP-SF-053")
@pytest.mark.req("SP-SF-059")
async def test_reads_are_immediate_before_during_and_after_model_work() -> None:
    """能力と状態と手元の声一覧は、モデルの操作を待たず、閉じた後も読める。"""
    engine = SampleTTSEngine()
    assert engine.health().state == "unprepared"
    assert not engine.health().busy
    assert engine.health().last_progress_monotonic_s is None
    assert engine.capabilities.languages == frozenset({"en", "ja"})
    assert engine.capabilities.auto_language is False
    assert engine.capabilities.streaming is True
    assert engine.capabilities.voice_listing is True
    assert engine.capabilities.voice_cloning is False
    assert engine.capabilities.sample_rate == 24000
    with pytest.raises(FrozenInstanceError):
        engine.capabilities.streaming = False
    engine.prepare_gate.clear()
    task = asyncio.create_task(engine.prepare())
    await engine.prepare_started.wait()
    assert engine.health().state == "unprepared"
    assert engine.health().busy
    assert engine.list_voices() == ["default"]
    assert not inspect.isawaitable(engine.capabilities)
    engine.prepare_gate.set()
    await task
    assert engine.health().state == "ready"
    assert not engine.health().busy
    await engine.aclose()
    assert engine.health().state == "closed"
    assert engine.list_voices() == ["default"]
    assert engine.capabilities.sample_rate == 24000


@pytest.mark.req("SP-SF-053")
async def test_model_dependent_voice_list_rejects_unprepared_and_closed_state() -> None:
    """モデルを読んで得る声一覧は用意の前と資源を返した後に状態の誤りを知らせる。"""

    class LoadedVoicesTTSEngine(SampleTTSEngine):
        def list_voices(self) -> list[str]:
            if self.health().state != "ready":
                raise error("EngineStateError")("Model voices require a prepared engine")
            return super().list_voices()

    engine = LoadedVoicesTTSEngine()
    with pytest.raises(error("EngineStateError")):
        engine.list_voices()
    await engine.prepare()
    assert engine.list_voices() == ["default"]
    await engine.aclose()
    with pytest.raises(error("EngineStateError")):
        engine.list_voices()


@pytest.mark.req("SP-SF-055")
async def test_prepare_loads_once_and_returns_during_synthesis() -> None:
    """用意は読み込みを一度だけ行い、用意済みなら合成の終了を待たず戻る。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    engine.synthesis_gate.clear()
    work = asyncio.create_task(engine.get("active"))
    await engine.synthesis_started.wait()
    again = await submitted(engine.prepare())
    try:
        assert again.done()
        await again
        assert engine.calls.count("prepare") == 1
    finally:
        engine.synthesis_gate.set()
        await work
        await again
        await engine.aclose()


@pytest.mark.req("SP-SF-055")
async def test_concurrent_prepare_waits_for_the_same_successful_loading() -> None:
    """読み込み中の用意は進行中の読み込みが成功するまで待ち、読み込みを重ねない。"""
    engine = SampleTTSEngine()
    engine.prepare_gate.clear()
    first = asyncio.create_task(engine.prepare())
    await engine.prepare_started.wait()
    second = await submitted(engine.prepare())
    try:
        assert not second.done()
        assert engine.calls == ["prepare"]
    finally:
        engine.prepare_gate.set()
        await asyncio.gather(first, second)
        await engine.aclose()
    assert engine.calls.count("prepare") == 1


@pytest.mark.req("SP-SF-055")
@pytest.mark.req("SP-SF-056")
async def test_failed_prepare_preserves_failure_and_allows_retry_without_release() -> None:
    """読み込みの失敗は原因を知らせて用意前に戻り、親は資源返却を呼ばず再び用意できる。"""
    engine = SampleTTSEngine()
    native = OSError("model cannot load")
    engine.prepare_failure = native
    with pytest.raises(error("InferenceError")) as caught:
        await engine.prepare()
    assert caught.value.__cause__ is native
    assert engine.health().state == "unprepared"
    assert not engine.health().busy
    assert "release" not in engine.calls
    engine.prepare_failure = None
    await engine.prepare()
    assert engine.calls.count("prepare") == 2
    await engine.aclose()
    assert engine.calls.count("release") == 1


@pytest.mark.req("SP-SF-055")
@pytest.mark.req("SP-SF-060")
async def test_prepare_reports_environment_failure_without_probing_operation() -> None:
    """環境の不足は用意の失敗で知らせ、元の原因を保つ。"""
    engine = SampleTTSEngine()
    native = OSError("CUDA library missing")
    failure = error("EnvironmentUnavailableError")("CUDA unavailable")
    failure.__cause__ = native
    engine.prepare_failure = failure
    with pytest.raises(error("EnvironmentUnavailableError")) as caught:
        await engine.prepare()
    assert caught.value is failure
    assert caught.value.__cause__ is native
    await engine.aclose()
    assert "release" not in engine.calls


@pytest.mark.req("SP-SF-054")
@pytest.mark.req("SP-SF-055")
def test_removed_engine_operations_and_errors_are_absent() -> None:
    """用意と終了の古い操作、engine の同期の口、古い error の種類を公開しない。"""
    for name in (
        "warmup",
        "wait_ready",
        "get_sync",
        "stream_sync",
        "warmup_sync",
        "wait_ready_sync",
        "close_sync",
        "resolve_ready_options",
        "ready_timeout_error",
        "DEFAULT_READY_TIMEOUT",
        "check_availability",
    ):
        assert not hasattr(TTSEngineBase, name), name
    for name in ("AudioProcessingError", "StreamingError", "EngineNotFoundError"):
        assert not hasattr(speechflow, name), name
        assert not hasattr(speechflow.core, name), name
        assert not hasattr(speechflow.core.exceptions, name), name


@pytest.mark.req("SP-SF-053")
@pytest.mark.parametrize("operation", ["get", "stream"])
async def test_synthesis_requires_preparation(operation: str) -> None:
    """合成は用意の前に状態の誤りで断り、モデルの仕事を始めない。"""
    engine = SampleTTSEngine()
    with pytest.raises(error("EngineStateError")):
        if operation == "get":
            await engine.get("text")
        else:
            await anext(engine.stream("text"))
    assert engine.calls == []
    await engine.aclose()


@pytest.mark.req("SP-SF-003")
async def test_get_combines_samples_and_forwards_engine_options() -> None:
    """一括の合成は音声を結合し、model と voice と engine 固有の引数を仕事へ渡す。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    result = await engine.get("hello", model="model", voice="voice", speed=2)
    assert np.array_equal(result.data, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert result.sample_rate == 24000
    assert result.channels == 1
    assert engine.arguments == ("hello", "model", "voice", {"speed": 2})
    assert engine.health().last_progress_monotonic_s is not None
    await engine.aclose()


@pytest.mark.req("SP-SF-003")
@pytest.mark.req("SP-SF-059")
async def test_stream_yields_incrementally_and_records_progress() -> None:
    """逐次の合成は次の chunk を待たず最初の chunk を返し、進行時刻を更新する。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    engine.synthesis_gate.clear()
    stream = engine.stream("hello", model="model", voice="voice", speed=3)
    first = await anext(stream)
    assert np.array_equal(first.data, engine.chunks[0])
    assert engine.arguments == ("hello", "model", "voice", {"speed": 3})
    snapshot = engine.health()
    assert snapshot.busy
    assert snapshot.last_progress_monotonic_s is not None
    engine.synthesis_gate.set()
    remaining = [chunk async for chunk in stream]
    assert len(remaining) == 1
    progress = engine.health().last_progress_monotonic_s
    assert progress is not None and progress >= snapshot.last_progress_monotonic_s
    assert not engine.health().busy
    await engine.aclose()


@pytest.mark.req("SP-SF-061")
@pytest.mark.parametrize("operation", ["get", "stream"])
@pytest.mark.parametrize("text", ["", " \n\t"])
async def test_empty_text_is_rejected_before_native_synthesis(operation: str, text: str) -> None:
    """空または空白だけの text は合成を始めず設定の誤りで断る。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    with pytest.raises(error("ConfigurationError")):
        if operation == "get":
            await engine.get(text)
        else:
            await anext(engine.stream(text))
    assert engine.calls == ["prepare", "prepared"]
    await engine.aclose()


@pytest.mark.req("SP-SF-061")
@pytest.mark.parametrize("operation", ["get", "stream"])
@pytest.mark.parametrize("sizes", [[], [0, 0], [0, 1, 0]])
async def test_empty_output_is_decided_by_total_samples(operation: str, sizes: list[int]) -> None:
    """正常に終わった合成は chunk の数ではなく sample の総数が零の時だけ空出力を知らせる。"""
    engine = SampleTTSEngine()
    engine.chunks = [np.zeros(size, dtype=np.float32) for size in sizes]
    await engine.prepare()

    async def consume() -> int:
        if operation == "get":
            return int((await engine.get("text")).data.shape[0])
        return sum([chunk.data.shape[0] async for chunk in engine.stream("text")])

    if sum(sizes):
        assert await consume() == sum(sizes)
    else:
        with pytest.raises(error("EmptyOutputError")):
            await consume()
    await engine.aclose()


@pytest.mark.req("SP-SF-060")
@pytest.mark.parametrize("operation", ["get", "stream"])
async def test_native_synthesis_failure_is_not_a_partial_success(operation: str) -> None:
    """合成途中の失敗は、作成済みの音声を正常な一括結果にせず原因を持つ error で知らせる。"""
    engine = SampleTTSEngine()
    native = OSError("native inference failed")
    engine.synthesis_failure = native
    await engine.prepare()
    with pytest.raises(error("InferenceError")) as caught:
        if operation == "get":
            await engine.get("text")
        else:
            async for _chunk in engine.stream("text"):
                pass
    assert caught.value.__cause__ is native
    assert not engine.health().busy
    await engine.aclose()


@pytest.mark.req("SP-SF-057")
async def test_stream_holds_its_turn_until_explicitly_closed_then_queue_is_fifo() -> None:
    """逐次の合成は chunk の間も順番を持ち、後始末の後に待つ合成を来た順で実行する。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    stream = engine.stream("first")
    await anext(stream)
    second = await submitted(engine.get("second"))
    third = await submitted(engine.get("third"))
    try:
        assert not second.done() and not third.done()
        assert engine.calls == ["prepare", "prepared", "first"]
        assert engine.health().busy
    finally:
        await stream.aclose()
        await asyncio.gather(second, third)
        await engine.aclose()
    assert engine.calls.index("finished:first") < engine.calls.index("second") < engine.calls.index("third")


@pytest.mark.req("SP-SF-057")
async def test_queued_synthesis_can_be_cancelled_without_starting_native_work() -> None:
    """順番を待つ合成は利用側が取り消せ、その仕事を始めず次の操作を残す。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    stream = engine.stream("first")
    await anext(stream)
    waiting = await submitted(engine.get("cancelled"))
    waiting.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert "cancelled" not in engine.calls
    finally:
        await stream.aclose()
        await engine.aclose()


@pytest.mark.req("SP-SF-055")
@pytest.mark.req("SP-SF-058")
async def test_caller_can_bound_prepare_and_loading_finishes_before_timeout_returns() -> None:
    """用意の時間の上限は利用側が付け、取り消した読み込みの終了を待ってから返る。"""
    engine = SampleTTSEngine()
    engine.prepare_gate.clear()
    engine.prepare_cleanup.clear()
    deadline = asyncio.get_running_loop().create_future()

    async def loading() -> None:
        async with asyncio.timeout(None) as limit:
            deadline.set_result(limit)
            await engine.prepare()

    task = asyncio.create_task(loading())
    await engine.prepare_started.wait()
    limit = await deadline
    limit.reschedule(asyncio.get_running_loop().time())
    await engine.prepare_cancelled.wait()
    try:
        assert not task.done()
        assert engine.health().busy
    finally:
        engine.prepare_cleanup.set()
        with pytest.raises(TimeoutError):
            await task
        await engine.aclose()


@pytest.mark.req("SP-SF-058")
@pytest.mark.parametrize("operation", ["get", "stream"])
async def test_caller_cancellation_waits_for_native_cleanup_and_accepts_next_call(operation: str) -> None:
    """利用側の取り消しは仕事の後始末を待ち、部分の結果を返さず次の合成を受け付ける。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    engine.synthesis_gate.clear()
    engine.synthesis_cleanup.clear()
    stream = engine.stream("cancelled")
    if operation == "get":
        task = asyncio.create_task(engine.get("cancelled"))
        await engine.synthesis_started.wait()
    else:
        await anext(stream)
        task = asyncio.ensure_future(anext(stream))
        await checkpoint()
    task.cancel()
    await engine.synthesis_cancelled.wait()
    next_task = await submitted(engine.get("next"))
    try:
        assert not task.done()
        assert not next_task.done()
        task.cancel()  # Repeated cancellation cannot cut native cleanup short.
        await checkpoint()
        assert not task.done()
        assert "next" not in engine.calls
    finally:
        engine.synthesis_cleanup.set()
        engine.synthesis_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await next_task
        await engine.aclose()
    assert engine.calls.index("finished:cancelled") < engine.calls.index("next")


@pytest.mark.req("SP-SF-058")
async def test_early_stream_close_waits_for_cleanup_before_next_model_operation() -> None:
    """列を最後まで読まず閉じた時は後始末を待ち、その後だけ次のモデルの操作を始める。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    stream = engine.stream("early")
    await anext(stream)
    engine.synthesis_cleanup.clear()
    close = asyncio.create_task(stream.aclose())
    await engine.synthesis_cancelled.wait()
    following = await submitted(engine.get("following"))
    try:
        assert not close.done()
        assert not following.done()
    finally:
        engine.synthesis_cleanup.set()
        await asyncio.gather(close, following)
        await engine.aclose()
    assert engine.calls.index("finished:early") < engine.calls.index("following")


@pytest.mark.req("SP-SF-056")
@pytest.mark.parametrize("operation", ["get", "stream", "prepare"])
async def test_close_cancels_active_and_waiting_work_before_returning_resources(operation: str) -> None:
    """終了は進行中と順番待ちの仕事を断り、仕事の終了を待ってから資源を返す。"""
    engine = SampleTTSEngine()
    active: asyncio.Future[Any]
    if operation == "prepare":
        engine.prepare_gate.clear()
        engine.prepare_cleanup.clear()
        engine.finish_prepare_on_cancel = True
        active = asyncio.create_task(engine.prepare())
        await engine.prepare_started.wait()
        cancelled = engine.prepare_cancelled
        cleanup = engine.prepare_cleanup
    else:
        await engine.prepare()
        engine.synthesis_gate.clear()
        engine.synthesis_cleanup.clear()
        if operation == "get":
            active = asyncio.create_task(engine.get("active"))
            await engine.synthesis_started.wait()
        else:
            stream = engine.stream("active")
            await anext(stream)
            active = asyncio.ensure_future(anext(stream))
            await checkpoint()
        cancelled = engine.synthesis_cancelled
        cleanup = engine.synthesis_cleanup
    queued = await submitted(engine.get("queued"))
    close = asyncio.create_task(engine.aclose())
    await cancelled.wait()
    try:
        with pytest.raises(error("EngineClosedError")):
            await queued
        assert "queued" not in engine.calls
        assert "release" not in engine.calls
        assert not close.done()
        assert not active.done()
        with pytest.raises(error("EngineClosedError")):
            await engine.prepare()
    finally:
        cleanup.set()
        with pytest.raises(error("EngineClosedError")):
            await active
        await close
    assert engine.calls[-2:] == ["release", "released"]
    assert engine.health().state == "closed"


@pytest.mark.req("SP-SF-056")
async def test_close_also_cleans_a_stream_suspended_between_chunks() -> None:
    """終了は利用側が次を求めていない逐次の合成も取り消し、資源を返す。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    stream = engine.stream("suspended")
    await anext(stream)
    await engine.aclose()
    assert engine.calls.index("finished:suspended") < engine.calls.index("release")
    with pytest.raises(error("EngineClosedError")):
        await anext(stream)
    await stream.aclose()


@pytest.mark.req("SP-SF-056")
async def test_close_after_cancelled_loading_waits_for_success_then_releases() -> None:
    """利用側が取り消した読み込みが実際には成功した時、終了はその終了の後で資源を返す。"""
    engine = SampleTTSEngine()
    engine.prepare_gate.clear()
    engine.prepare_cleanup.clear()
    engine.finish_prepare_on_cancel = True
    prepare = asyncio.create_task(engine.prepare())
    await engine.prepare_started.wait()
    prepare.cancel()
    await engine.prepare_cancelled.wait()
    close = await submitted(engine.aclose())
    try:
        assert not close.done()
        assert "release" not in engine.calls
    finally:
        engine.prepare_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await prepare
        await close
    assert engine.calls == ["prepare", "prepared", "release", "released"]


@pytest.mark.req("SP-SF-056")
async def test_overlapping_and_repeated_close_release_only_once() -> None:
    """重なった終了は一回の資源返却を待ち、閉じた後の終了は何もせず、仕事をする操作は用意し直さずに断る。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    engine.release_gate.clear()
    first = asyncio.create_task(engine.aclose())
    await engine.release_started.wait()
    second = await submitted(engine.aclose())
    try:
        assert not second.done()
        assert engine.calls.count("release") == 1
    finally:
        engine.release_gate.set()
        await asyncio.gather(first, second)
    await engine.aclose()
    assert engine.calls.count("release") == 1
    for operation in (engine.prepare(), engine.get("text"), anext(engine.stream("text")), engine.__aenter__()):
        with pytest.raises(error("EngineClosedError")):
            await operation


@pytest.mark.req("SP-SF-056")
async def test_close_failure_keeps_cause_and_does_not_release_again() -> None:
    """資源返却の失敗は原因を知らせて閉じた状態になり、二度目の終了では返却を繰り返さない。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    native = OSError("resource release failed")
    engine.release_failure = native
    with pytest.raises(error("InferenceError")) as caught:
        await engine.aclose()
    assert caught.value.__cause__ is native
    assert engine.health().state == "closed"
    await engine.aclose()
    assert engine.calls.count("release") == 1


@pytest.mark.req("SP-SF-056")
async def test_cancelling_close_waits_for_resource_release() -> None:
    """終了を利用側が取り消しても資源返却を最後まで待ち、返却を繰り返さない。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    engine.release_gate.clear()
    close = asyncio.create_task(engine.aclose())
    await engine.release_started.wait()
    close.cancel()
    await checkpoint()
    try:
        assert not close.done()
    finally:
        engine.release_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await close
    await engine.aclose()
    assert engine.calls.count("release") == 1


@pytest.mark.req("SP-SF-056")
async def test_async_context_prepares_and_closes_even_on_body_failure() -> None:
    """async with は入る時に用意し、本文が失敗しても抜ける時に終了する。"""
    engine = SampleTTSEngine()
    with pytest.raises(LookupError, match="body failed"):
        async with engine as entered:
            assert entered is engine
            assert engine.health().state == "ready"
            raise LookupError("body failed")
    assert engine.calls == ["prepare", "prepared", "release", "released"]


@pytest.mark.req("SP-SF-046")
def test_prepared_unclosed_engine_does_not_prevent_interpreter_exit() -> None:
    """用意した engine を閉じなくても Python interpreter は終了する。"""
    script = """
import asyncio
from speechflow import AudioData, EngineCapabilities, TTSEngineBase
from speechflow.core.sync import run_sync
class Engine(TTSEngineBase):
    capabilities = EngineCapabilities(frozenset(), False, False, False, False, None)
    async def _prepare(self):
        await asyncio.to_thread(lambda: None)
    async def _synthesize(self, text, model=None, voice=None, **kwargs):
        yield AudioData(None, 24000)
    async def _release(self):
        raise AssertionError("close was not requested")
engine = Engine()
run_sync(engine.prepare())
assert engine.health().state == "ready"
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_version_fallback_when_version_file_missing() -> None:
    """Importing without the generated version uses the fallback version."""
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = f"""
import importlib.abc
import sys
class MissingVersion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "speechflow._version":
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, MissingVersion())
sys.path.insert(0, {source_root!r})
import speechflow
assert speechflow.__version__ == "0.0.0+unknown", speechflow.__version__
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-SF-058")
@pytest.mark.req("SP-SF-060")
async def test_explicit_stream_close_reports_native_cleanup_failure() -> None:
    """列を途中で閉じた後始末の失敗は、元の原因を持つ実行時の error で知らせる。"""
    native = OSError("stream cleanup failed")

    class CleanupFailureTTSEngine(SampleTTSEngine):
        async def _synthesize(
            self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
        ) -> AsyncIterator[AudioData]:
            try:
                yield AudioData(np.zeros(1, dtype=np.float32), 24000)
            finally:
                raise native

    engine = CleanupFailureTTSEngine()
    await engine.prepare()
    stream = engine.stream("text")
    await anext(stream)
    with pytest.raises(error("InferenceError")) as caught:
        await stream.aclose()
    assert caught.value.__cause__ is native
    await engine.aclose()


@pytest.mark.req("SP-SF-058")
async def test_cancelling_early_stream_close_waits_for_cleanup_and_preserves_cancellation() -> None:
    """列を途中で閉じる処理の取り消しは後始末を待ち、利用側の取り消しをそのまま知らせる。"""
    engine = SampleTTSEngine()
    await engine.prepare()
    stream = engine.stream("early")
    await anext(stream)
    engine.synthesis_cleanup.clear()
    close = asyncio.create_task(stream.aclose())
    await engine.synthesis_cancelled.wait()
    close.cancel()
    await checkpoint()
    try:
        assert not close.done()
    finally:
        engine.synthesis_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await close
        await engine.aclose()


class LateChunkTTSEngine(SampleTTSEngine):
    """Native inference completes with audio after receiving cancellation."""

    def __init__(self, *, initial_chunk: bool = False) -> None:
        super().__init__()
        self.initial_chunk = initial_chunk
        self.native_started = asyncio.Event()
        self.native_cancelled = asyncio.Event()
        self.native_finished = asyncio.Event()
        self.chunk_produced = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.cleanup_allowed = asyncio.Event()
        self.cleanup_done = asyncio.Event()
        self.read_after_cancellation = asyncio.Event()
        self.trailing_read_allowed = asyncio.Event()
        self.work_task: asyncio.Task[Any] | None = None

    async def _native_inference(self) -> AudioData:
        self.native_started.set()
        try:
            await self.native_finished.wait()
        except asyncio.CancelledError:
            self.native_cancelled.set()
            await self.native_finished.wait()
        self.chunk_produced.set()
        return AudioData(np.array([0.9], dtype=np.float32), 24000)

    async def _synthesize(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncIterator[AudioData]:
        if text != "cancelled":
            async for chunk in super()._synthesize(text, model=model, voice=voice, **kwargs):
                yield chunk
            return
        self.calls.append(text)
        self.work_task = asyncio.current_task()
        native = asyncio.create_task(self._native_inference())
        try:
            if self.initial_chunk:
                yield AudioData(np.array([0.1], dtype=np.float32), 24000)
            yield await native
            if self.native_cancelled.is_set():
                self.read_after_cancellation.set()
                await self.trailing_read_allowed.wait()
        finally:
            if not native.done() and not native.cancelling():
                native.cancel()
            await native
            self.cleanup_started.set()
            await self.cleanup_allowed.wait()
            self.cleanup_done.set()
            self.calls.append("finished:cancelled")


@pytest.mark.req("SP-SF-058")
@pytest.mark.req("SP-SF-056")
@pytest.mark.parametrize("trigger", ["stream_read", "stream_close", "get", "engine_close"])
async def test_cancellation_discards_late_native_chunk_and_finishes_cleanup(trigger: str) -> None:
    """取り消し後に推論が音声を返しても、残りの読み取りをせず後始末を待ち、正常な結果にしない。"""
    engine = LateChunkTTSEngine(initial_chunk=trigger == "stream_close")
    await engine.prepare()
    stream = engine.stream("cancelled")
    read: asyncio.Future[Any] | None = None
    following: asyncio.Task[AudioData] | None = None
    if trigger == "get":
        read = asyncio.create_task(engine.get("cancelled"))
    elif trigger == "stream_close":
        assert np.array_equal((await anext(stream)).data, np.array([0.1], dtype=np.float32))
    else:
        read = asyncio.ensure_future(anext(stream))
    await engine.native_started.wait()

    actor: asyncio.Future[Any]
    if trigger == "stream_close":
        actor = asyncio.create_task(stream.aclose())
    elif trigger == "engine_close":
        actor = asyncio.create_task(engine.aclose())
    else:
        assert read is not None
        read.cancel()
        actor = read
    await engine.native_cancelled.wait()
    if trigger != "engine_close":
        following = await submitted(engine.get("following"))
    try:
        assert not actor.done()
        assert engine.health().busy
        assert not engine.cleanup_started.is_set()
        engine.native_finished.set()
        await engine.chunk_produced.wait()
        await checkpoint()
        await checkpoint()
        assert engine.cleanup_started.is_set(), "Core waited for another read after the cancelled inference ended"
        assert not engine.read_after_cancellation.is_set(), "Core read more synthesis after receiving a cancelled chunk"
        assert not actor.done()
        assert "release" not in engine.calls
        if following is not None:
            assert not following.done()
        engine.cleanup_allowed.set()
        if trigger in {"get", "stream_read"}:
            with pytest.raises(asyncio.CancelledError):
                await actor
        else:
            await actor
        assert engine.cleanup_done.is_set()
        if trigger == "engine_close":
            assert read is not None
            with pytest.raises(speechflow.EngineClosedError):
                await read
            assert engine.calls.index("finished:cancelled") < engine.calls.index("release")
        elif following is not None:
            await following
            assert engine.calls.index("finished:cancelled") < engine.calls.index("following")
    finally:
        engine.native_finished.set()
        engine.cleanup_allowed.set()
        engine.trailing_read_allowed.set()
        # Red leaves the core suspended after the hook's final yield. This
        # extra cancellation is teardown only, after the behavior assertions.
        if engine.work_task is not None and not engine.work_task.done():
            engine.work_task.cancel()
        pending = [task for task in (actor, read, following) if task is not None]
        await asyncio.gather(*pending, return_exceptions=True)
        await stream.aclose()
        await engine.aclose()
