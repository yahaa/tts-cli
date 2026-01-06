"""Tests for cli module."""

import pytest
from tts_cli.cli import create_argument_parser


class TestArgumentParser:
    """Tests for CLI argument parser."""

    def test_text_argument(self):
        """--text argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "Hello world"])
        assert args.text == "Hello world"
        assert args.input_file is None

    def test_file_argument(self):
        """--file argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--file", "input.txt"])
        assert args.input_file == "input.txt"
        assert args.text is None

    def test_input_argument(self):
        """--input argument should work (alias for --file)."""
        parser = create_argument_parser()
        args = parser.parse_args(["--input", "input.txt"])
        assert args.input_file == "input.txt"

    def test_output_argument(self):
        """--output argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--output", "out.wav"])
        assert args.output_audio == "out.wav"

    def test_output_audio_argument(self):
        """--output-audio argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--output-audio", "out.wav"])
        assert args.output_audio == "out.wav"

    def test_subtitle_argument(self):
        """--subtitle argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--subtitle", "out.srt"])
        assert args.output_srt == "out.srt"

    def test_output_srt_argument(self):
        """--output-srt argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--output-srt", "out.srt"])
        assert args.output_srt == "out.srt"

    def test_default_output(self):
        """Default output should be output.wav."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test"])
        assert args.output_audio == "output.wav"

    def test_speed_argument(self):
        """--speed argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--speed", "5"])
        assert args.speed == 5

    def test_default_speed(self):
        """Default speed should be 3."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test"])
        assert args.speed == 3

    def test_language_argument(self):
        """--language argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--language", "zh"])
        assert args.language == "zh"

    def test_default_language(self):
        """Default language should be en."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test"])
        assert args.language == "en"

    def test_speaker_argument(self):
        """--speaker argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--speaker", "voice.pt"])
        assert args.speaker == "voice.pt"

    def test_save_speaker_argument(self):
        """--save-speaker argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--save-speaker", "voice.pt"])
        assert args.save_speaker == "voice.pt"

    def test_max_length_argument(self):
        """--max-length argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--max-length", "800"])
        assert args.max_length == 800

    def test_whisper_model_argument(self):
        """--whisper-model argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--whisper-model", "small"])
        assert args.whisper_model == "small"

    def test_skip_subtitles_flag(self):
        """--skip-subtitles flag should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--skip-subtitles"])
        assert args.skip_subtitles is True

    def test_no_normalize_flag(self):
        """--no-normalize flag should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--no-normalize"])
        assert args.no_normalize is True

    def test_quiet_flag(self):
        """--quiet flag should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--quiet"])
        assert args.quiet is True

    def test_break_level_argument(self):
        """--break-level argument should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--break-level", "3"])
        assert args.break_level == 3

    def test_no_json_flag(self):
        """--no-json flag should work."""
        parser = create_argument_parser()
        args = parser.parse_args(["--text", "test", "--no-json"])
        assert args.no_json is True

    def test_require_input(self):
        """Either --text or --file is required."""
        parser = create_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
