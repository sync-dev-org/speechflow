# SpeechFlow

[English](https://github.com/sync-dev-org/speechflow/blob/main/README.md) | 日本語

SpeechFlow は text-to-speech (TTS) のための非同期 Python library です。core は engine に
共通する約束、音声データ、error、非同期の操作を同期の code から呼ぶ関数を提供します。
engine module は独立した distribution です。core の依存は NumPy だけで、optional extra は
設けません。

[Source repository](https://github.com/sync-dev-org/speechflow)

## インストール

SpeechFlow は Python 3.12 と Python 3.13 に対応します。core は PyPI から導入します。

```bash
pip install speechflow
```

uv の project へ追加する場合:

```bash
uv add speechflow
```

core は声の model や合成 engine を含みません。各 engine module は、自身の導入方法、
対応 platform、model の要件を文書化します。

## Quick Start

この例は core とその依存の NumPy だけで動きます。engine の API を示すために無音の chunk を
二つ返す例であり、text を声へ変換しません。実際の module は `_prepare()` で model の読み込みと
試しの推論を行い、`_synthesize()` で音声を返し、`_release()` で資源を返します。

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

`async with` は block に入る前に engine を用意し、block を抜ける時に閉じます。
`get()` は出力をまとめて返し、`stream()` は `AudioData` の chunk を逐次返します。
以降の例は、この節の `ExampleTTSEngine` を使います。

## Engine の用意と終了

合成の前に engine を用意します。用意の操作は model の読み込みと試しの推論を一度行い、
用意済みの engine に対する再呼び出しは直ちに戻ります。用意に失敗した engine は未準備のまま
残り、次の呼び出しで再試行できます。時間の上限は利用側が選び、例えば `asyncio.timeout()` で
付けます。

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

一つの engine instance は、用意、合成、終了を通して一つの event loop の上で使います。
閉じた instance は再利用できないので、新しい instance を作ります。`aclose()` は待機中と
実行中の model の操作を取り消し、処理と後始末の終了を待ってから資源を返します。閉じる操作は
何度呼んでも安全です。利用側による取り消しも、内部の処理の終了を待ちます。
core は処理が止まる速さを約束しません。

一つの engine の model の操作は、到着順に一度に一つだけ実行します。
`engine.capabilities` と `engine.health()` は、model の処理を待たずに同期で読めます。
health の値は `state` (`unprepared`、`ready`、`closed`)、`busy`、
`last_progress_monotonic_s` を持ちます。

### Stream の後始末

途中まで読んだ stream は明示的に閉じ、後始末が終わってから engine を再利用します。
`contextlib.aclosing` は async loop からの早期終了を扱えます。

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

### 同期の Code から呼ぶ

`speechflow.run_sync()` は、一つの共有 daemon 背景 event loop で coroutine を実行し、
結果を待ちます。`speechflow.iterate_sync()` は、同じ loop で async iterator を読みます。
呼び出し元が処理を止めた場合も、両関数は後始末を待ちます。途中まで読んだ同期 iterator は、
例えば `contextlib.closing` を使って明示的に閉じます。

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

この engine のすべての操作は共有の背景 loop で呼びます。application 自身の loop で操作する時は、
別の engine instance を使います。これらの関数は別の thread から呼べ、呼び出し元に実行中の
event loop がある場合も動きます。非同期の code では通常 `await` と `async for` を使います。
共有の背景 loop 自身から両関数を呼ぶと `EngineStateError` を送出します。

`speechflow.iterate_async(iterable)` は、同期 iterable を別の thread で読み、呼び出し元の
event loop を塞ぎません。早期終了と取り消しは、実行中の読み取りの終了を待ち、閉じる操作を持つ
入力を閉じます。

## Public API

package root と `speechflow.core` は次の名前を公開します。

| Name | 用途 |
| --- | --- |
| `AudioData` | PCM sample と音声 metadata |
| `TTSEngineBase` | 用意、合成、取り消し、終了の共通の動き |
| `EngineCapabilities` | 言語、streaming、声、出力 rate の不変の申告 |
| `EngineState` | `UNPREPARED`、`READY`、`CLOSED` の状態 |
| `HealthSnapshot` | 直ちに読める状態、busy の値、最終進捗時刻 |
| `run_sync`, `iterate_sync`, `iterate_async` | 非同期と同期の code をつなぐ関数 |
| `TTSError` | engine error の基底 |
| `ConfigurationError` | 入力や構成の不正 |
| `EngineStateError` | engine の状態に適合しない操作 |
| `EngineClosedError` | 閉じた engine の使用や、閉じる操作による取り消し |
| `InferenceError` | model の操作や資源の解放の失敗 |
| `EmptyOutputError` | 空でない text から音声 sample が得られなかった場合 |
| `EnvironmentUnavailableError` | 現在の環境では用意を実行できない場合 |
| `ModuleNotInstalledError` | 登録済み engine module が未導入の場合 |

`speechflow.__version__` は core distribution の version を返します。engine class は各
module root が公開します。`get(text, model=None, voice=None)` と
`stream(text, model=None, voice=None)` は基底の引数を保ち、module 固有の option は追加の
keyword argument として受け取ります。

### 能力の申告と声の一覧

`EngineCapabilities` は `languages`、`auto_language`、`streaming`、`voice_listing`、
`voice_cloning`、`sample_rate` を持ちます。`sample_rate` が `None` の場合は、model を
読み込むまで出力 rate が分かりません。返された各 `AudioData` が実際の rate を持つので、
固定の rate を仮定せず、その値を使います。

`voice_listing=True` の module は、同期の `list_voices()` を持ちます。戻り値は
`list[dict[str, str]]` で、各要素は `{"name": <声の名前>, "kind": <声の種類>}` の形です。
`voice_listing=False` の module はこの操作を持ちません。core の基底 class もこの操作を
定義しません。

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

engine の作者は、有限値で `[-1, 1]` に正規化した float32 sample を返し、通常は mono とし、
Hz 単位の `sample_rate` を明示します。`data` は書き込み可能で、engine 内部の buffer と
共有しない独立した copy でなければなりません。mono の shape は `[samples]`、複数 channel の
shape は `[samples, channels]` です。`channels` の既定値は `1`、`format` は `"pcm"` です。

`AudioData` は値を保持する container です。構築時に配列の検証、正規化、copy を行いません。
この約束を満たす出力を作る責務は engine module が持ちます。

## Error

engine の失敗をまとめて扱う場合は `TTSError`、個別に回復する場合は subclass を捕捉します。

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

不正な text や空白だけの text は `ConfigurationError`、用意前の合成は `EngineStateError`、
終了後の使用は `EngineClosedError` を送出します。音声 sample がないまま完了した応答や
stream は `EmptyOutputError` を送出します。model の操作と資源の解放の失敗は、元の例外を
原因として持つ `InferenceError` へ変換されます。既に `TTSError` の体系にある engine の
error はそのまま送出されます。

core の登録済み module 表は現在空です。engine module は、その module root から明示的に
import します。未導入の module の明示的な import では、Python が `ModuleNotFoundError` を
送出します。package の登録済みの属性だけが、module の欠損時に `ModuleNotInstalledError` を
使います。未知の属性は `AttributeError` を送出します。

## Application の責務

application は設定、credential、model weights とその取得、外部 process の管理、deployment を
担います。core は `.env` file を読みません。音声 file への書き出し、resampling、effect、
echo cancellation、noise suppression、音声認識、対話の進行は core の範囲外です。

## ドキュメント

[ドキュメントのガイド](https://sync-dev-org.github.io/speechflow/) から、engine の用意と終了、
同期の code から呼ぶ関数、音声データ、error の API reference を参照できます。

## 開発

[公開用の repository](https://github.com/sync-dev-org/speechflow) は core の source と test を
持ちます。検証には engine module の checkout、GPU、model weights、native toolchain を
必要としません。

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

各 engine module の repository は、自身の metadata、lock、source、test、build、配布を
持ちます。core の repository はそれらの distribution を収集したり build したりしません。

## License

[MIT](https://github.com/sync-dev-org/speechflow/blob/main/LICENSE)
