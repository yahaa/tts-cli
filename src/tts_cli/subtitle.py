"""Subtitle generation using Whisper for tts-cli."""

import json
from typing import Dict, Optional

from .utils import format_timestamp, print_info, print_success


def generate_srt_file(whisper_result: Dict, output_path: str) -> None:
    """
    Generate SRT subtitle file from Whisper result.

    Args:
        whisper_result: Whisper transcription result
        output_path: Output SRT file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(whisper_result["segments"], 1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n")
            f.write("\n")


def generate_srt_from_segments(
    segments: list,
    output_path: str
) -> None:
    """
    Generate SRT file from a list of segments.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys
        output_path: Output SRT file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n")
            f.write("\n")


def generate_subtitles(
    audio_path: str,
    language: str,
    whisper_model_size: str,
    output_srt_path: str,
    output_json_path: Optional[str] = None,
    generate_json: bool = False,
    quiet: bool = False
) -> Dict:
    """
    Generate SRT subtitles using Whisper.

    Args:
        audio_path: Path to audio file
        language: Language code ('en' or 'zh')
        whisper_model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        output_srt_path: Output SRT file path
        output_json_path: Output JSON file path (optional)
        generate_json: Whether to generate JSON output
        quiet: Suppress progress messages

    Returns:
        Whisper transcription result dict
    """
    import whisper

    if not quiet:
        print_info(f"Language: {language}")
        print_info(f"Loading Whisper model '{whisper_model_size}'...")

    # Load Whisper model
    model = whisper.load_model(whisper_model_size)

    if not quiet:
        print_info("Recognizing audio...")

    # Transcribe audio
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    if not quiet:
        print_success("Recognition complete")
        preview_text = result['text'][:100] + '...' if len(result['text']) > 100 else result['text']
        print_info(f"Recognized text: {preview_text}")

    # Generate SRT file
    generate_srt_file(result, output_srt_path)

    if not quiet:
        print_success(f"SRT subtitle saved: {output_srt_path}")

    # Generate JSON file (optional)
    if generate_json and output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if not quiet:
            print_success(f"JSON data saved: {output_json_path}")

    return result


def parse_srt_file(srt_path: str) -> list:
    """
    Parse an SRT file and return a list of segments.

    Args:
        srt_path: Path to SRT file

    Returns:
        List of dicts with 'index', 'start', 'end', 'text' keys
    """
    segments = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines (segment separator)
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = int(lines[0])
            time_line = lines[1]
            text = '\n'.join(lines[2:])

            # Parse time
            start_str, end_str = time_line.split(' --> ')
            start = _parse_timestamp(start_str)
            end = _parse_timestamp(end_str)

            segments.append({
                'index': index,
                'start': start,
                'end': end,
                'text': text
            })

    return segments


def _parse_timestamp(ts: str) -> float:
    """
    Parse SRT timestamp to seconds.

    Args:
        ts: Timestamp string (HH:MM:SS,mmm)

    Returns:
        Time in seconds
    """
    # Handle both comma and period as decimal separator
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds
