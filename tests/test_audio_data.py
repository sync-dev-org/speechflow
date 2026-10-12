"""Unit tests for the AudioData container.

AC: foundation.S7.AC4
"""

import numpy as np
import pytest

from speechflow import AudioData


class TestAudioData:
    """Tests for the AudioData dataclass container."""

    @pytest.mark.req("SP-SF-003")
    def test_explicit_field_creation(self) -> None:
        """Test that all fields are set correctly when provided explicitly.

        AC: foundation.S3.AC1
        """
        data = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        audio = AudioData(data=data, sample_rate=44100, channels=2, format="wav")

        assert np.array_equal(audio.data, data)
        assert audio.sample_rate == 44100
        assert audio.channels == 2
        assert audio.format == "wav"

    @pytest.mark.req("SP-SF-003")
    def test_default_channels_and_format(self) -> None:
        """Test that channels defaults to 1 and format defaults to 'pcm'.

        AC: foundation.S3.AC1
        """
        data = np.zeros(1000, dtype=np.float32)
        audio = AudioData(data=data, sample_rate=22050)

        assert audio.channels == 1
        assert audio.format == "pcm"

    @pytest.mark.req("SP-SF-003")
    def test_numpy_array_dtype_is_float32(self) -> None:
        """Test that the data array dtype is float32.

        AC: foundation.S3.AC1
        """
        data = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        audio = AudioData(data=data, sample_rate=44100)

        assert audio.data.dtype == np.float32
