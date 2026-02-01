"""Tests for utils module."""

import pytest
import tempfile
import os

from tts_cli.utils import (
    format_timestamp,
    validate_speed,
    validate_language,
    read_text_from_file,
)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_zero_seconds(self):
        """Zero seconds should format correctly."""
        assert format_timestamp(0) == "00:00:00,000"

    def test_simple_seconds(self):
        """Simple seconds should format correctly."""
        assert format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        """Minutes should format correctly."""
        assert format_timestamp(125.75) == "00:02:05,750"

    def test_hours(self):
        """Hours should format correctly."""
        assert format_timestamp(3661.123) == "01:01:01,123"

    def test_large_value(self):
        """Large values should format correctly."""
        assert format_timestamp(7200) == "02:00:00,000"


class TestValidateSpeed:
    """Tests for validate_speed function."""

    def test_valid_speed_0(self):
        """Speed 0 should be valid."""
        assert validate_speed(0) == 0

    def test_valid_speed_9(self):
        """Speed 9 should be valid."""
        assert validate_speed(9) == 9

    def test_valid_speed_middle(self):
        """Middle speed should be valid."""
        assert validate_speed(5) == 5

    def test_invalid_speed_negative(self):
        """Negative speed should raise ValueError."""
        with pytest.raises(ValueError, match="Speed must be between 0-9"):
            validate_speed(-1)

    def test_invalid_speed_too_high(self):
        """Speed > 9 should raise ValueError."""
        with pytest.raises(ValueError, match="Speed must be between 0-9"):
            validate_speed(10)


class TestValidateLanguage:
    """Tests for validate_language function (returns Qwen3-TTS language names)."""

    def test_valid_english(self):
        """English should map to 'english'."""
        assert validate_language("en") == "english"

    def test_valid_chinese(self):
        """Chinese should map to 'chinese'."""
        assert validate_language("zh") == "chinese"

    def test_valid_auto(self):
        """Auto should map to 'auto'."""
        assert validate_language("auto") == "auto"

    def test_valid_japanese(self):
        """Japanese should map to 'japanese'."""
        assert validate_language("ja") == "japanese"

    def test_invalid_language(self):
        """Invalid language should raise ValueError."""
        with pytest.raises(ValueError, match="Language must be one of"):
            validate_language("xx")


class TestReadTextFromFile:
    """Tests for read_text_from_file function."""

    def test_read_utf8_file(self):
        """UTF-8 file should be read correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Hello, world!")
            temp_path = f.name

        try:
            result = read_text_from_file(temp_path)
            assert result == "Hello, world!"
        finally:
            os.unlink(temp_path)

    def test_read_chinese_text(self):
        """Chinese text should be read correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("你好，世界！")
            temp_path = f.name

        try:
            result = read_text_from_file(temp_path)
            assert result == "你好，世界！"
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_text_from_file("/nonexistent/path/file.txt")

    def test_read_utf8_bom_file(self):
        """UTF-8 with BOM should be read correctly."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Write UTF-8 BOM + content
            f.write(b'\xef\xbb\xbfHello with BOM')
            temp_path = f.name

        try:
            result = read_text_from_file(temp_path)
            assert "Hello with BOM" in result
        finally:
            os.unlink(temp_path)
