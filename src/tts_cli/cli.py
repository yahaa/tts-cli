"""Command-line interface for tts-cli."""

import argparse
import sys

from .core import TTSConfig, run_tts_with_subtitles
from .utils import validate_speed, validate_language


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="ChatTTS Text-to-Speech CLI with Subtitle Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "Hello, world!"
  %(prog)s --file article.txt --output output.wav --subtitle output.srt
  %(prog)s --input story.txt --output-audio story.wav --speed 2
  %(prog)s --text "Quick test" --skip-subtitles
  %(prog)s --file data.txt --quiet
        """
    )

    # Input group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Direct text input to convert to speech"
    )
    # Support both --file (README) and --input (reference implementation)
    input_group.add_argument(
        "--file",
        type=str,
        dest="input_file",
        help="Read text from file (UTF-8 encoded)"
    )
    input_group.add_argument(
        "--input",
        type=str,
        dest="input_file",
        help="Read text from file (UTF-8 encoded) - alias for --file"
    )

    # Output options - support both naming conventions
    parser.add_argument(
        "--output", "--output-audio",
        type=str,
        dest="output_audio",
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--subtitle", "--output-srt",
        type=str,
        dest="output_srt",
        default=None,
        help="Output SRT subtitle file path (default: auto-derived from audio name)"
    )

    # Audio generation options
    parser.add_argument(
        "--speed",
        type=int,
        default=3,
        help="Speech speed 0-9 (default: 3)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Language code: 'en' or 'zh' (default: en)"
    )
    parser.add_argument(
        "--break-level",
        type=int,
        default=5,
        choices=range(8),
        help="Punctuation pause strength 0-7 (default: 5, higher=longer pauses)"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Speaker embedding file (PT format). If not provided, a random speaker will be sampled"
    )
    parser.add_argument(
        "--save-speaker",
        type=str,
        default=None,
        help="Save the used speaker embedding to file for reuse"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Auto-split text longer than this into chunks (default: None, recommended: 800)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (normalization is enabled by default)"
    )

    # Subtitle options
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Generate audio only, skip subtitle generation"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output generation"
    )

    # Output control
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages, show only file paths"
    )

    return parser


def main():
    """CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Validate parameters
        validate_speed(args.speed)
        validate_language(args.language)

        # Create config from args
        config = TTSConfig(
            text=args.text,
            input_file=args.input_file,
            output_audio=args.output_audio,
            output_srt=args.output_srt,
            speed=args.speed,
            language=args.language,
            break_level=args.break_level,
            speaker=args.speaker,
            save_speaker=args.save_speaker,
            max_length=args.max_length,
            no_normalize=args.no_normalize,
            whisper_model=args.whisper_model,
            skip_subtitles=args.skip_subtitles,
            no_json=args.no_json,
            quiet=args.quiet,
        )

        # Run main workflow
        run_tts_with_subtitles(config)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
