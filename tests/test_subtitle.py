"""Tests for subtitle module."""

import pytest
import tempfile
import os

from tts_cli.subtitle import (
    generate_srt_file,
    generate_srt_from_segments,
    parse_srt_file,
    _parse_timestamp,
)
from tts_cli.utils import format_timestamp


class TestGenerateSrtFile:
    """Tests for generate_srt_file function."""

    def test_generate_simple_srt(self):
        """Simple SRT should be generated correctly."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": "How are you"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = f.name

        try:
            generate_srt_file(whisper_result, temp_path)

            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert "1\n" in content
            assert "00:00:00,000 --> 00:00:02,500" in content
            assert "Hello world" in content
            assert "2\n" in content
            assert "How are you" in content
        finally:
            os.unlink(temp_path)

    def test_generate_chinese_srt(self):
        """Chinese text SRT should be generated correctly."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "你好世界"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = f.name

        try:
            generate_srt_file(whisper_result, temp_path)

            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert "你好世界" in content
        finally:
            os.unlink(temp_path)


class TestGenerateSrtFromSegments:
    """Tests for generate_srt_from_segments function."""

    def test_generate_from_list(self):
        """SRT should be generated from segment list."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "First"},
            {"start": 1.0, "end": 2.0, "text": "Second"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = f.name

        try:
            generate_srt_from_segments(segments, temp_path)

            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert "First" in content
            assert "Second" in content
        finally:
            os.unlink(temp_path)


class TestParseSrtFile:
    """Tests for parse_srt_file function."""

    def test_parse_simple_srt(self):
        """Simple SRT should be parsed correctly."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:02,500 --> 00:00:05,000
How are you
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            temp_path = f.name

        try:
            segments = parse_srt_file(temp_path)

            assert len(segments) == 2
            assert segments[0]['index'] == 1
            assert segments[0]['start'] == 0.0
            assert segments[0]['end'] == 2.5
            assert segments[0]['text'] == "Hello world"
            assert segments[1]['index'] == 2
            assert segments[1]['text'] == "How are you"
        finally:
            os.unlink(temp_path)

    def test_roundtrip(self):
        """Generate and parse should produce same data."""
        original_segments = [
            {"start": 0.0, "end": 1.5, "text": "Test one"},
            {"start": 1.5, "end": 3.0, "text": "Test two"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = f.name

        try:
            generate_srt_from_segments(original_segments, temp_path)
            parsed = parse_srt_file(temp_path)

            assert len(parsed) == len(original_segments)
            for orig, parsed_seg in zip(original_segments, parsed):
                assert abs(orig['start'] - parsed_seg['start']) < 0.001
                assert abs(orig['end'] - parsed_seg['end']) < 0.001
                assert orig['text'] == parsed_seg['text']
        finally:
            os.unlink(temp_path)


class TestParseTimestamp:
    """Tests for _parse_timestamp function."""

    def test_parse_zero(self):
        """Zero timestamp should parse to 0.0."""
        assert _parse_timestamp("00:00:00,000") == 0.0

    def test_parse_seconds(self):
        """Seconds should parse correctly."""
        assert _parse_timestamp("00:00:05,500") == 5.5

    def test_parse_minutes(self):
        """Minutes should parse correctly."""
        assert _parse_timestamp("00:02:30,000") == 150.0

    def test_parse_hours(self):
        """Hours should parse correctly."""
        assert _parse_timestamp("01:00:00,000") == 3600.0

    def test_parse_with_period(self):
        """Period separator should also work."""
        assert _parse_timestamp("00:00:05.500") == 5.5


class TestFormatTimestampRoundtrip:
    """Test format and parse roundtrip."""

    def test_roundtrip_values(self):
        """Format and parse should roundtrip correctly."""
        test_values = [0.0, 1.5, 60.0, 3661.123, 7200.5]

        for value in test_values:
            formatted = format_timestamp(value)
            parsed = _parse_timestamp(formatted)
            assert abs(value - parsed) < 0.001, f"Roundtrip failed for {value}"
