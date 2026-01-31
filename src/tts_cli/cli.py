"""Command-line interface for tts-cli."""

import argparse
import os
import sys

from .core import TTSConfig, VoiceConfig, run_tts_with_subtitles
from .utils import SUPPORTED_LANGUAGES, validate_language, validate_speed
from .voice import SUPPORTED_SPEAKERS


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the generate command."""
    # Input group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str, help="Direct text input to convert to speech"
    )
    # Support both --file (README) and --input (reference implementation)
    input_group.add_argument(
        "--file",
        type=str,
        dest="input_file",
        help="Read text from file (UTF-8 encoded)",
    )
    input_group.add_argument(
        "--input",
        type=str,
        dest="input_file",
        help="Read text from file (UTF-8 encoded) - alias for --file",
    )

    # Output options - support both naming conventions
    parser.add_argument(
        "--output",
        "--output-audio",
        type=str,
        dest="output_audio",
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    parser.add_argument(
        "--subtitle",
        "--output-srt",
        type=str,
        dest="output_srt",
        default=None,
        help="Output SRT subtitle file path (default: auto-derived from audio name)",
    )

    # Audio generation options
    parser.add_argument(
        "--speed", type=int, default=3, help="Speech speed 0-9 (default: 3)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=SUPPORTED_LANGUAGES,
        help=f"Language: {', '.join(SUPPORTED_LANGUAGES)} (default: auto)",
    )
    parser.add_argument(
        "--break-level",
        type=int,
        default=5,
        choices=range(8),
        help="Punctuation pause strength 0-7 (default: 5, higher=longer pauses)",
    )

    # Voice options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["custom", "design", "clone"],
        help="Voice mode: custom (preset speakers), design (description), clone (reference audio)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Voice file (.qwen-voice) OR preset name (Ryan, Vivian, Serena, etc.). Default: Ryan",
    )
    parser.add_argument(
        "--save-speaker",
        type=str,
        default=None,
        help="Save voice to file (.qwen-voice format)",
    )
    parser.add_argument(
        "--voice-description",
        type=str,
        help="Natural language voice description (for --mode design)",
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        help="Reference audio for voice cloning (3+ seconds, WAV/MP3/FLAC)",
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        help="Transcript of reference audio",
    )

    # Processing options
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Auto-split text longer than this into chunks (default: 1000)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="Max chunks to process in parallel (default: 4)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (normalization is enabled by default)",
    )

    # Subtitle options
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Skip subtitle generation (default: True)",
    )
    parser.add_argument(
        "--subtitles",
        action="store_true",
        default=False,
        help="Enable subtitle generation (requires openai-whisper)",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="Skip JSON output generation"
    )

    # Output control
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages, show only file paths",
    )


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the serve command."""
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--mongodb-uri",
        type=str,
        default="mongodb://localhost:27017",
        help="MongoDB connection URI (default: mongodb://localhost:27017)",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="tts-server",
        help="MongoDB database name (default: tts-server)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tts_output",
        help="Directory for generated files (default: ./tts_output)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of background workers (default: 1)",
    )
    parser.add_argument(
        "--cleanup-hours",
        type=int,
        default=24,
        help="Hours to keep completed tasks (default: 24)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Text-to-Speech CLI with Voice Cloning and Subtitle Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with preset voice
  %(prog)s --text "Hello, world!" --speaker Ryan

  # Voice cloning
  %(prog)s --mode clone --reference-audio my_voice.wav --reference-text "Sample" --text "Hello"

  # From file
  %(prog)s --file article.txt --output output.wav --subtitle output.srt

  # Long text with batch processing
  %(prog)s --file story.txt --max-length 1000 --max-batch 4

  # Skip subtitles
  %(prog)s --text "Quick test" --skip-subtitles

  # Server mode
  %(prog)s serve --port 8000 --mongodb-uri mongodb://localhost:27017
        """,
    )

    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command (explicit)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate TTS audio (default command)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_generate_arguments(generate_parser)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start HTTP API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_serve_arguments(serve_parser)

    # For backward compatibility, also add generate arguments to main parser
    # This allows `tts-cli --text "Hello"` to work without the generate subcommand
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--text", type=str, help="Direct text input to convert to speech"
    )
    input_group.add_argument(
        "--file",
        type=str,
        dest="input_file",
        help="Read text from file (UTF-8 encoded)",
    )
    input_group.add_argument(
        "--input",
        type=str,
        dest="input_file",
        help="Read text from file (UTF-8 encoded) - alias for --file",
    )

    # Output options - support both naming conventions
    parser.add_argument(
        "--output",
        "--output-audio",
        type=str,
        dest="output_audio",
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    parser.add_argument(
        "--subtitle",
        "--output-srt",
        type=str,
        dest="output_srt",
        default=None,
        help="Output SRT subtitle file path (default: auto-derived from audio name)",
    )

    # Audio generation options
    parser.add_argument(
        "--speed", type=int, default=3, help="Speech speed 0-9 (default: 3)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=SUPPORTED_LANGUAGES,
        help=f"Language: {', '.join(SUPPORTED_LANGUAGES)} (default: auto)",
    )
    parser.add_argument(
        "--break-level",
        type=int,
        default=5,
        choices=range(8),
        help="Punctuation pause strength 0-7 (default: 5, higher=longer pauses)",
    )

    # Voice options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["custom", "design", "clone"],
        help="Voice mode: custom (preset speakers), design (description), clone (reference audio)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Voice file (.qwen-voice) OR preset name (Ryan, Vivian, Serena, etc.). Default: Ryan",
    )
    parser.add_argument(
        "--save-speaker",
        type=str,
        default=None,
        help="Save voice to file (.qwen-voice format)",
    )
    parser.add_argument(
        "--voice-description",
        type=str,
        help="Natural language voice description (for --mode design)",
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        help="Reference audio for voice cloning (3+ seconds, WAV/MP3/FLAC)",
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        help="Transcript of reference audio",
    )

    # Processing options
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Auto-split text longer than this into chunks (default: 1000)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="Max chunks to process in parallel (default: 4)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (normalization is enabled by default)",
    )

    # Subtitle options
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Skip subtitle generation (default: True)",
    )
    parser.add_argument(
        "--subtitles",
        action="store_true",
        default=False,
        help="Enable subtitle generation (requires openai-whisper)",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="Skip JSON output generation"
    )

    # Output control
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages, show only file paths",
    )

    return parser


def _run_generate(args) -> None:
    """Run the generate command."""
    # Validate parameters
    validate_speed(args.speed)
    args.language = validate_language(args.language)

    # Determine voice mode
    voice_mode = "custom"
    if hasattr(args, "mode") and args.mode:
        voice_mode = args.mode
    elif hasattr(args, "reference_audio") and args.reference_audio:
        voice_mode = "clone"
    elif hasattr(args, "voice_description") and args.voice_description:
        voice_mode = "design"

    # Parse speaker argument (file path vs preset name)
    speaker_file = None
    speaker_name = "Ryan"  # Default speaker

    if hasattr(args, "speaker") and args.speaker:
        # Check if it's a file or preset name
        if args.speaker.endswith(".qwen-voice") or args.speaker.endswith(".pt"):
            speaker_file = args.speaker
        elif args.speaker in SUPPORTED_SPEAKERS:
            speaker_name = args.speaker
        else:
            # Try as file path
            if os.path.exists(args.speaker):
                speaker_file = args.speaker
            else:
                # Assume it's a preset name and let validation handle it
                speaker_name = args.speaker

    # Map speed to instruct parameter
    instruct = None
    if args.speed <= 2:
        instruct = "speak slowly and clearly"
    elif args.speed >= 7:
        instruct = "speak quickly"

    # Create voice config
    voice_config = VoiceConfig(
        mode=voice_mode,
        speaker=speaker_name,
        voice_description=getattr(args, "voice_description", None),
        reference_audio=getattr(args, "reference_audio", None),
        reference_text=getattr(args, "reference_text", None),
        voice_prompt_file=speaker_file,
        save_voice_prompt_path=args.save_speaker,
        instruct=instruct,
    )

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
        max_batch=args.max_batch,
        no_normalize=args.no_normalize,
        whisper_model=args.whisper_model,
        skip_subtitles=not args.subtitles,  # Default: skip subtitles unless --subtitles specified
        no_json=args.no_json,
        quiet=args.quiet,
        voice=voice_config,
    )

    # Run main workflow
    run_tts_with_subtitles(config)


def _run_serve(args) -> None:
    """Run the serve command."""
    try:
        from .server import run_server
    except ImportError as e:
        print(
            "Error: Server dependencies not installed. "
            "Install with: pip install tts-cli[server]",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    run_server(
        host=args.host,
        port=args.port,
        mongodb_uri=args.mongodb_uri,
        db_name=args.db_name,
        output_dir=args.output_dir,
        num_workers=args.workers,
        cleanup_hours=args.cleanup_hours,
        log_level=args.log_level,
    )


def main():
    """CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Determine which command to run
        if args.command == "serve":
            _run_serve(args)
        elif args.command == "generate":
            _run_generate(args)
        elif args.text or args.input_file:
            # Backward compatibility: no subcommand but has text/file input
            _run_generate(args)
        else:
            # No command and no input specified, show help
            parser.print_help()
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
