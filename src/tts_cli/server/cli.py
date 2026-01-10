"""Command-line interface for tts-server."""

import argparse
import sys


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="TTS Server - HTTP API for Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --port 9000
  %(prog)s --mongodb-uri mongodb://db:27017 --db-name my-tts
  %(prog)s --workers 2 --output-dir /data/tts
        """,
    )

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

    return parser


def main():
    """CLI entry point for tts-server."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        from . import run_server

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
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
