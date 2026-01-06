"""
tts-cli: A local offline text-to-speech CLI tool with subtitle generation.

This package provides a command-line tool for converting text to speech using
ChatTTS, with optional subtitle generation using Whisper.
"""

__version__ = "0.1.0"

from .core import TTSConfig, run_tts_with_subtitles
from .text_processor import (
    split_text_intelligently,
    split_paragraph_to_sentences,
    normalize_text_for_tts,
    detect_language,
)
from .audio import (
    float_to_int16,
    merge_audio_files,
    merge_with_pauses,
    normalize_audio,
    get_audio_duration,
    SAMPLE_RATE,
)
from .subtitle import (
    generate_subtitles,
    generate_srt_file,
    parse_srt_file,
)
from .utils import (
    format_timestamp,
    validate_speed,
    validate_language,
    read_text_from_file,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "TTSConfig",
    "run_tts_with_subtitles",
    # Text processing
    "split_text_intelligently",
    "split_paragraph_to_sentences",
    "normalize_text_for_tts",
    "detect_language",
    # Audio
    "float_to_int16",
    "merge_audio_files",
    "merge_with_pauses",
    "normalize_audio",
    "get_audio_duration",
    "SAMPLE_RATE",
    # Subtitle
    "generate_subtitles",
    "generate_srt_file",
    "parse_srt_file",
    # Utils
    "format_timestamp",
    "validate_speed",
    "validate_language",
    "read_text_from_file",
]
