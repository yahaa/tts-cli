"""Tests for audio module."""

import pytest
import numpy as np

from tts_cli.audio import (
    float_to_int16,
    merge_audio_files,
    merge_with_pauses,
    normalize_audio,
    get_audio_duration,
    SAMPLE_RATE,
)


class TestFloatToInt16:
    """Tests for float_to_int16 function."""

    def test_normalize_small_audio(self):
        """Small amplitude audio should be normalized."""
        audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        result = float_to_int16(audio)
        assert result.dtype == np.int16
        # Should be scaled up
        assert np.max(np.abs(result)) > 1000

    def test_normalize_full_range(self):
        """Full range audio should remain in valid range."""
        audio = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result = float_to_int16(audio)
        assert result.dtype == np.int16
        assert np.max(result) <= 32767
        assert np.min(result) >= -32768

    def test_single_value(self):
        """Single value should be handled."""
        audio = np.array([0.5], dtype=np.float32)
        result = float_to_int16(audio)
        assert result.dtype == np.int16


class TestMergeAudioFiles:
    """Tests for merge_audio_files function."""

    def test_merge_empty_list(self):
        """Empty list should return empty array."""
        result = merge_audio_files([])
        assert len(result) == 0

    def test_merge_single_array(self):
        """Single array should be returned as is."""
        audio = np.array([1, 2, 3], dtype=np.int16)
        result = merge_audio_files([audio])
        np.testing.assert_array_equal(result, audio)

    def test_merge_two_arrays(self):
        """Two arrays should be merged with silence."""
        audio1 = np.array([1, 2], dtype=np.int16)
        audio2 = np.array([3, 4], dtype=np.int16)
        result = merge_audio_files([audio1, audio2], sample_rate=1000, pause_duration=0.01)

        # Should have audio1 + silence + audio2
        assert len(result) > len(audio1) + len(audio2)

    def test_merge_custom_pause(self):
        """Custom pause duration should work."""
        audio1 = np.array([1], dtype=np.int16)
        audio2 = np.array([2], dtype=np.int16)
        result = merge_audio_files([audio1, audio2], sample_rate=1000, pause_duration=1.0)

        # 1 sample + 1000 silence samples + 1 sample = 1002
        assert len(result) == 1002


class TestMergeWithPauses:
    """Tests for merge_with_pauses function."""

    def test_empty_segments(self):
        """Empty segments should return empty array."""
        result = merge_with_pauses([], [0])
        assert len(result) == 0

    def test_single_paragraph_single_sentence(self):
        """Single sentence should be returned as is."""
        audio = np.array([1, 2, 3], dtype=np.int16)
        result = merge_with_pauses([audio], [0, 1])
        np.testing.assert_array_equal(result, audio)

    def test_two_sentences_same_paragraph(self):
        """Two sentences in same paragraph should have short pause."""
        audio1 = np.array([1, 2], dtype=np.int16)
        audio2 = np.array([3, 4], dtype=np.int16)
        result = merge_with_pauses(
            [audio1, audio2],
            [0, 2],  # One paragraph with 2 sentences
            sample_rate=1000,
            sentence_pause=0.01,
            paragraph_pause=0.02
        )
        # Should have sentence pause (10 samples) between
        assert len(result) == 2 + 10 + 2


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_one_second(self):
        """24000 samples at 24kHz should be 1 second."""
        audio = np.zeros(24000, dtype=np.int16)
        assert get_audio_duration(audio) == 1.0

    def test_half_second(self):
        """12000 samples at 24kHz should be 0.5 seconds."""
        audio = np.zeros(12000, dtype=np.int16)
        assert get_audio_duration(audio) == 0.5

    def test_empty_audio(self):
        """Empty audio should be 0 seconds."""
        audio = np.array([], dtype=np.int16)
        assert get_audio_duration(audio) == 0.0


class TestNormalizeAudio:
    """Tests for normalize_audio function."""

    def test_normalize_preserves_dtype(self):
        """Normalize should return int16 dtype."""
        audio = np.array([1000, -1000, 500], dtype=np.int16)
        result = normalize_audio(audio)
        # Should return int16 array
        assert result.dtype == np.int16
        # Should preserve relative values
        assert len(result) == len(audio)

    def test_empty_audio(self):
        """Empty audio should return empty."""
        audio = np.array([], dtype=np.int16)
        result = normalize_audio(audio)
        assert len(result) == 0
