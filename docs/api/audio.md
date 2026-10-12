# Audio data API

`AudioData` is available directly from `speechflow` and from `speechflow.core`.
It holds PCM samples and metadata for complete responses and streaming chunks.
Engine authors return finite float32 samples normalized to `[-1, 1]`, normally
mono, with an explicit `sample_rate` in Hz.

The `data` array must be writable and independent of the engine's internal
buffers. Mono data has shape `[samples]`; multichannel data has shape
`[samples, channels]`. `channels` defaults to `1` and `format` to `"pcm"`.
When `engine.capabilities.sample_rate` is `None`, each returned `AudioData`
still carries its actual sample rate.

`AudioData` is a plain container. It does not validate, normalize, or copy arrays
at construction time. The engine module is responsible for producing output that
meets these requirements. See the [guide's example](../index.md#audiodata).

::: speechflow.core.base.AudioData
