"""Utility functions for tts-cli."""

from pathlib import Path
from typing import Dict, List


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def validate_speed(speed: int) -> int:
    """
    Validate speed parameter is in range 0-9.

    Args:
        speed: Speed value to validate

    Returns:
        Validated speed value

    Raises:
        ValueError: If speed is out of range
    """
    if not 0 <= speed <= 9:
        raise ValueError(f"Speed must be between 0-9, got {speed}")
    return speed


def validate_language(lang: str) -> str:
    """
    Validate language code.

    Args:
        lang: Language code to validate

    Returns:
        Validated language code

    Raises:
        ValueError: If language code is not supported
    """
    if lang not in ["en", "zh"]:
        raise ValueError(f"Language must be 'en' or 'zh', got '{lang}'")
    return lang


def read_text_from_file(filepath: str) -> str:
    """
    Read text content from file with UTF-8 encoding.

    Args:
        filepath: Path to text file

    Returns:
        Text content

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is not a file
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {filepath}")

    # Try UTF-8 first
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try UTF-8 with BOM
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return f.read()


def check_dependencies(require_whisper: bool = True) -> None:
    """
    Check availability of required dependencies.

    Args:
        require_whisper: Whether Whisper is required

    Raises:
        ImportError: If required dependencies are missing
    """
    # ChatTTS check
    try:
        import ChatTTS  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ChatTTS not found. Install it with:\n"
            "   pip install ChatTTS"
        ) from exc

    if require_whisper:
        try:
            import whisper  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Whisper not found. You have two options:\n\n"
                "1. Install Whisper to enable subtitle generation:\n"
                "   pip install openai-whisper\n\n"
                "2. Skip subtitle generation and generate audio only:\n"
                "   tts-cli --text \"...\" --skip-subtitles"
            ) from exc


# ========================================
# Progress Output Functions
# ========================================

def print_header() -> None:
    """Print CLI header."""
    print("=" * 70)
    print("ChatTTS Text-to-Speech with Subtitle Generation")
    print("=" * 70)


def print_step(step_num: int, total_steps: int, description: str) -> None:
    """Print step header."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")


def print_info(message: str, indent: int = 1) -> None:
    """Print indented info message."""
    print("   " * indent + message)


def print_success(message: str, indent: int = 1) -> None:
    """Print success message."""
    print(f"{'   ' * indent}[OK] {message}")


def print_summary(stats: Dict) -> None:
    """Print final summary."""
    print(f"\n[Statistics]")
    print(f"   Audio duration: {stats['duration']:.2f} seconds")
    if stats['segments'] > 0:
        print(f"   Subtitle segments: {stats['segments']} segments")
    print(f"   Total characters: {stats['characters']} characters")
    print(f"   Average speed: {stats['chars_per_sec']:.1f} chars/sec")

    print("\n" + "=" * 70)
    print("All done!")
    print("=" * 70)
    print("\nGenerated files:")
    for i, filepath in enumerate(stats['files'], 1):
        print(f"   {i}. {filepath}")

    if stats['segments'] > 0:
        print("\nTip: Use a video player to load the subtitle file")
    print("=" * 70 + "\n")


def print_final_paths_only(files: List[str]) -> None:
    """Print only file paths for scripting purposes."""
    for filepath in files:
        print(filepath)
