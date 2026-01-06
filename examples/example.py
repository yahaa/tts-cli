"""
ChatTTS Text-to-Speech CLI with Subtitle Generation

A command-line tool for generating speech audio and synchronized SRT subtitles
from text input using ChatTTS for speech synthesis and Whisper for transcription.

Dependencies:
    - ChatTTS: TTS model
    - openai-whisper: Speech recognition (optional with --skip-subtitles)
    - scipy: Audio file I/O
    - numpy: Audio processing
    - torch: Deep learning framework

Usage Examples:
    # Basic usage with direct text
    python tts_cli.py --text "Hello, world!"

    # Read from file
    python tts_cli.py --input article.txt --speed 2

    # Long text with auto-splitting for better quality
    python tts_cli.py --input long_article.txt --max-length 800 --save-speaker voice.pt

    # Reuse saved voice for consistency
    python tts_cli.py --input chapter2.txt --speaker voice.pt

    # Custom output paths
    python tts_cli.py --input story.txt --output-audio story.wav

    # Audio only (skip subtitles)
    python tts_cli.py --text "Quick test" --skip-subtitles

    # Quiet mode for automation
    python tts_cli.py --input data.txt --quiet

    # Disable text normalization (keep numbers and special chars as-is)
    python tts_cli.py --text "Raw text 2025" --no-normalize

For detailed help:
    python tts_cli.py --help
"""

import argparse
import sys
import json
import re
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Suppress Whisper Triton warning on Windows (Triton not supported, falls back to slower DTW)
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")

import numpy as np
import torch
import scipy.io.wavfile as wavfile
import ChatTTS
import math

# Try to import numba for JIT optimization
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorator if numba is not available
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator


# ========================================
# Audio Normalization
# ========================================

@jit(nopython=True)
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Adaptive float to int16 conversion with normalization.
    Normalizes audio to maximize use of int16 range without clipping.

    Args:
        audio: Audio data as float array (typically -1.0 to 1.0)

    Returns:
        Normalized audio as int16 array
    """
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


# ========================================
# GPU Memory Utils
# ========================================

def get_gpu_free_memory_mb() -> float:
    """
    Get available GPU memory in MB.

    Returns:
        Available GPU memory in MB, or 0 if no GPU available.
    """
    if not torch.cuda.is_available():
        return 0

    try:
        torch.cuda.synchronize()
        free_memory = torch.cuda.mem_get_info()[0]  # Returns (free, total)
        return free_memory / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0


def estimate_memory_per_chunk_mb(chunk_chars: int = 800, fp16: bool = True) -> float:
    """
    Estimate GPU memory usage per chunk based on ChatTTS model architecture.

    Based on GPT config: hidden_size=768, num_hidden_layers=20, max_new_token=2048

    Args:
        chunk_chars: Number of characters per chunk
        fp16: Whether using FP16 (default) or FP32

    Returns:
        Estimated memory usage in MB per chunk
    """
    # Rough token estimation: ~2 tokens per character
    input_tokens = chunk_chars * 2
    output_tokens = 2048  # max_new_token
    total_tokens = input_tokens + output_tokens

    hidden_size = 768
    num_layers = 20
    bytes_per_element = 2 if fp16 else 4

    # KV cache: 2 (K+V) * layers * seq_len * hidden_size * bytes
    kv_cache_mb = (2 * num_layers * total_tokens * hidden_size * bytes_per_element) / (1024 * 1024)

    # Intermediate activations (roughly 1.5x KV cache)
    activations_mb = kv_cache_mb * 1.5

    # Add safety margin (1.3x)
    total_mb = (kv_cache_mb + activations_mb) * 1.3

    return total_mb


def calculate_optimal_batch_size(
    num_chunks: int,
    chunk_chars: int = 800,
    reserved_memory_mb: float = 1024,
    min_batch: int = 1,
    max_batch: int = 6
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    Args:
        num_chunks: Total number of chunks to process
        chunk_chars: Average characters per chunk
        reserved_memory_mb: Memory to reserve for model and system (default 1GB)
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size
    """
    free_memory = get_gpu_free_memory_mb()

    if free_memory <= 0:
        # No GPU or can't detect, use conservative batch size
        return min(2, num_chunks)

    # Available memory for inference
    available_mb = free_memory - reserved_memory_mb
    if available_mb <= 0:
        return min_batch

    # Memory per chunk
    mem_per_chunk = estimate_memory_per_chunk_mb(chunk_chars)

    # Calculate batch size
    batch_size = int(available_mb / mem_per_chunk)

    # Clamp to valid range
    batch_size = max(min_batch, min(batch_size, max_batch, num_chunks))

    return batch_size


# ========================================
# Utility Functions
# ========================================

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
        UnicodeDecodeError: If file encoding is not UTF-8
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


def normalize_text_for_tts(text: str) -> str:
    """
    Normalize text to avoid ChatTTS invalid character warnings.

    This function converts numbers, special punctuation, and other
    characters that ChatTTS doesn't handle well into TTS-friendly text.

    Args:
        text: Input text

    Returns:
        Normalized text safe for ChatTTS
    """
    import re

    # Number to word mapping for common cases
    num_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

    # Replace common year patterns first (before general number replacement)
    text = text.replace('2025', 'twenty twenty-five')
    text = text.replace('2024', 'twenty twenty-four')
    text = text.replace('2023', 'twenty twenty-three')

    # Replace time patterns (e.g., "5:30" -> "five thirty")
    def replace_time(match):
        hour = match.group(1)
        minute = match.group(2)
        hour_word = num_to_word.get(hour, hour)
        minute_word = num_to_word.get(minute, minute)
        return f"{hour_word} {minute_word}"

    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', replace_time, text)

    # Replace standalone single digits with words
    def replace_digit(match):
        digit = match.group(0)
        return num_to_word.get(digit, digit)

    text = re.sub(r'\b\d\b', replace_digit, text)

    # Replace multi-digit numbers in common patterns
    # e.g., "10" -> "ten", "25" -> "twenty-five"
    def replace_number(match):
        num = int(match.group(0))
        if num <= 20:
            words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                    'nineteen', 'twenty']
            return words[num] if num < len(words) else str(num)
        return str(num)  # Keep larger numbers as is for now

    text = re.sub(r'\b\d{1,2}\b', replace_number, text)

    # Replace dashes and special punctuation
    text = text.replace('—', ', ')  # em-dash
    text = text.replace('–', ', ')  # en-dash
    text = text.replace(':', ', ')  # colon
    text = text.replace(';', ', ')  # semicolon

    # Replace quotes with simple quotes or remove
    text = text.replace('"', '')
    text = text.replace('"', '')
    text = text.replace(''', "'")
    text = text.replace(''', "'")

    # Clean up multiple spaces and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove backslashes
    text = text.replace('\\', '')

    return text.strip()


def split_text_intelligently(text: str, max_length: int = 800, min_tail_length: int = 300) -> List[str]:
    """
    Split long text into smaller chunks by paragraphs.

    Strategy:
    1. First split by paragraphs (one or more newlines)
    2. If a paragraph exceeds 800 characters, split at the last period (.)
       that keeps the chunk under 800 characters
    3. If the last chunk is too short (< min_tail_length), merge it with the previous chunk

    Args:
        text: Input text to split
        max_length: Maximum length of each chunk (default: 800)
        min_tail_length: Minimum length for the last chunk (default: 300)
                        If shorter, merge with previous chunk

    Returns:
        List of text chunks
    """
    # If text is short enough, return as is
    if len(text) <= max_length:
        return [text]

    # Step 1: Split by paragraphs (one or more newlines)
    paragraphs = re.split(r'\n+', text)

    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph is short enough, add directly
        if len(para) <= max_length:
            chunks.append(para)
            continue

        # Step 2: Paragraph too long, split at last period within limit
        remaining = para
        while len(remaining) > max_length:
            # Find the last period within max_length
            search_text = remaining[:max_length]
            last_period = search_text.rfind('.')

            if last_period > 0:
                # Split at the last period
                chunk = remaining[:last_period + 1].strip()
                remaining = remaining[last_period + 1:].strip()
                chunks.append(chunk)
            else:
                # No period found, force split at max_length
                chunks.append(remaining[:max_length].strip())
                remaining = remaining[max_length:].strip()

        # Add remaining text
        if remaining:
            chunks.append(remaining)

    # Step 3: Merge short tail chunk with previous chunk to avoid generation issues
    # Short chunks can cause the model to immediately generate end tokens
    if len(chunks) > 1 and len(chunks[-1]) < min_tail_length:
        last_chunk = chunks.pop()
        chunks[-1] = chunks[-1] + " " + last_chunk

    return chunks


def split_paragraph_to_sentences(text: str) -> List[str]:
    """
    Split a paragraph into sentences.

    Args:
        text: Input paragraph text

    Returns:
        List of sentences
    """
    # Split at sentence endings (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def merge_audio_files(audio_arrays: List[np.ndarray], sample_rate: int = 24000, pause_duration: float = 0.7) -> np.ndarray:
    """
    Merge multiple audio arrays into a single audio file.

    Args:
        audio_arrays: List of audio numpy arrays
        sample_rate: Sample rate of the audio
        pause_duration: Duration of silence between segments in seconds (default: 0.7)

    Returns:
        Merged audio array
    """
    if len(audio_arrays) == 0:
        return np.array([], dtype=np.int16)
    if len(audio_arrays) == 1:
        return audio_arrays[0]

    # Add silence between segments
    silence_samples = int(sample_rate * pause_duration)
    silence = np.zeros(silence_samples, dtype=np.int16)

    # Merge all arrays with silence in between
    merged = audio_arrays[0]
    for audio in audio_arrays[1:]:
        merged = np.concatenate([merged, silence, audio])

    return merged


def check_dependencies(require_whisper: bool = True) -> None:
    """
    Check availability of required dependencies.

    Args:
        require_whisper: Whether Whisper is required

    Raises:
        ImportError: If required dependencies are missing
    """
    # ChatTTS is already imported at module level

    if require_whisper:
        try:
            import whisper  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Whisper not found. You have two options:\n\n"
                "1. Install Whisper to enable subtitle generation:\n"
                "   pip install openai-whisper\n\n"
                "2. Skip subtitle generation and generate audio only:\n"
                "   python tts_cli.py --text \"...\" --skip-subtitles"
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
    """Print final summary matching tools.py style."""
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


def print_final_paths_only(files: list) -> None:
    """Print only file paths for scripting purposes."""
    for filepath in files:
        print(filepath)


# ========================================
# Core Processing Functions
# ========================================

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


def generate_audio(
    chat: ChatTTS.Chat,
    text: str,
    speed: int,
    output_path: str,
    speaker_file: Optional[str] = None,
    save_speaker_file: Optional[str] = None,
    break_level: int = 5,
    quiet: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Generate audio using ChatTTS.

    Args:
        chat: Initialized ChatTTS Chat instance
        text: Input text to convert to speech
        speed: Speech speed (0-9)
        output_path: Output audio file path
        speaker_file: Optional speaker embedding file to load
        save_speaker_file: Optional file path to save speaker embedding
        quiet: Suppress progress messages

    Returns:
        Tuple of (audio_data, duration_seconds)
    """
    if not quiet:
        print_info(f"Speed parameter: speed_{speed}")

    # Load or sample speaker embedding
    if speaker_file:
        if not quiet:
            print_info(f"Loading speaker from: {speaker_file}")
        spk = torch.load(speaker_file, weights_only=True)
    else:
        if not quiet:
            print_info("Sampling random speaker...")
        spk = chat.sample_random_speaker()

    # Save speaker if requested
    if save_speaker_file:
        torch.save(spk, save_speaker_file)
        if not quiet:
            print_info(f"Speaker saved to: {save_speaker_file}")

    if not quiet:
        print_info("Generating speech...")

    # Refine parameters for controlling pause/break at punctuation
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt=f"[break_{break_level}]",
        show_tqdm=not quiet,
    )

    # Audio generation parameters (based on tools.py and examples/cmd/run.py)
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk,  # CRITICAL: Speaker embedding for voice quality
        prompt=f"[speed_{speed}]",
        temperature=0.3,
        top_P=0.7,
        top_K=20,
        repetition_penalty=1.05,
        max_new_token=2048,
        ensure_non_empty=True,
    )

    # Generate audio
    wavs = chat.infer(
        [text],
        skip_refine_text=False,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        do_text_normalization=False,
        do_homophone_replacement=False,
        split_text=True,
        max_split_batch=10,
    )

    # Save audio with adaptive normalization
    audio_data = float_to_int16(wavs[0])
    wavfile.write(output_path, 24000, audio_data)

    duration = len(wavs[0]) / 24000

    if not quiet:
        print_success(f"Audio saved: {output_path}")
        print_info(f"Duration: {duration:.2f} seconds")

    return audio_data, duration


def generate_subtitles(
    audio_path: str,
    language: str,
    whisper_model_size: str,
    output_srt_path: str,
    output_json_path: Optional[str],
    generate_json: bool,
    quiet: bool = False
) -> Dict:
    """
    Generate SRT subtitles using Whisper.

    Args:
        audio_path: Path to audio file
        language: Language code
        whisper_model_size: Whisper model size
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
        print_info(f"Recognized text: {result['text'][:100]}...")

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


# ========================================
# Main Workflow
# ========================================

def run_tts_with_subtitles(args) -> None:
    """
    Main workflow orchestrator.

    Args:
        args: Parsed command-line arguments
    """
    # 1. Validate inputs
    if args.text and args.input:
        raise ValueError("Cannot specify both --text and --input")
    if not args.text and not args.input:
        raise ValueError("Must specify either --text or --input")

    # 2. Check dependencies
    require_whisper = not args.skip_subtitles
    check_dependencies(require_whisper)

    # 3. Read text input
    if args.input:
        text = read_text_from_file(args.input)
    else:
        text = args.text

    # Validate text is not empty
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    # Normalize text by default (unless disabled)
    if not args.no_normalize:
        original_length = len(text)
        text = normalize_text_for_tts(text)
        if not args.quiet:
            print_info(f"Text normalized ({original_length} → {len(text)} chars)")

    # 4. Print header (if not quiet)
    if not args.quiet:
        print_header()
        print_info(f"Input text length: {len(text)} characters")

    # 5. Initialize ChatTTS
    if not args.quiet:
        print_info("Loading ChatTTS model...")

    chat = ChatTTS.Chat()
    # Try to load from huggingface cache first, fall back to local download
    import os
    hf_cache_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub/models--2Noise--ChatTTS/snapshots")
    )
    if os.path.exists(hf_cache_path):
        # Find the latest snapshot
        snapshots = [d for d in os.listdir(hf_cache_path) if os.path.isdir(os.path.join(hf_cache_path, d))]
        if snapshots:
            custom_path = os.path.join(hf_cache_path, snapshots[0])
            if not args.quiet:
                print_info(f"Loading from cache: {custom_path}")
            chat.load(source="custom", custom_path=custom_path, compile=False)
        else:
            chat.load(source="huggingface", compile=False)
    else:
        chat.load(source="huggingface", compile=False)

    # 6. Check if text needs splitting
    text_chunks = []
    if args.max_length and len(text) > args.max_length:
        text_chunks = split_text_intelligently(text, args.max_length)
        if not args.quiet:
            print_info(f"Text split into {len(text_chunks)} chunks for better quality")
    else:
        text_chunks = [text]

    # 7. Generate audio (with or without splitting)
    total_steps = 2 if not args.skip_subtitles else 1

    if len(text_chunks) == 1:
        # Single chunk - direct generation
        if not args.quiet:
            print_step(1, total_steps, "Generating audio...")

        _, duration = generate_audio(
            chat=chat,
            text=text_chunks[0],
            speed=args.speed,
            output_path=args.output_audio,
            speaker_file=args.speaker,
            save_speaker_file=args.save_speaker,
            break_level=args.break_level,
            quiet=args.quiet
        )
    else:
        # Multiple paragraphs - process with sentence-level granularity
        # Pause rules: 1.0s between paragraphs, 0.5s between sentences
        num_paragraphs = len(text_chunks)

        # Split each paragraph into sentences
        all_sentences = []
        paragraph_boundaries = [0]  # Track where each paragraph starts

        for para in text_chunks:
            sentences = split_paragraph_to_sentences(para)
            if not sentences:
                sentences = [para]  # Fallback: treat whole paragraph as one sentence
            all_sentences.extend(sentences)
            paragraph_boundaries.append(len(all_sentences))

        num_sentences = len(all_sentences)
        avg_sentence_chars = sum(len(s) for s in all_sentences) // max(num_sentences, 1)

        # Calculate optimal batch size based on GPU memory
        batch_size = calculate_optimal_batch_size(
            num_chunks=num_sentences,
            chunk_chars=avg_sentence_chars
        )

        if not args.quiet:
            free_mem = get_gpu_free_memory_mb()
            mem_per_chunk = estimate_memory_per_chunk_mb(avg_sentence_chars)
            print_step(1, total_steps, f"Generating audio ({num_paragraphs} paragraphs, {num_sentences} sentences)...")
            print_info(f"GPU free memory: {free_mem:.0f} MB")
            print_info(f"Estimated memory per sentence: {mem_per_chunk:.0f} MB")
            print_info(f"Auto batch size: {batch_size}")

        # Load or create speaker (ensure consistency across chunks)
        if args.speaker:
            spk = torch.load(args.speaker, weights_only=True)
        else:
            spk = chat.sample_random_speaker()
            if args.save_speaker:
                torch.save(spk, args.save_speaker)
                if not args.quiet:
                    print_info(f"Speaker saved to: {args.save_speaker}")

        if not args.quiet:
            for i, para in enumerate(text_chunks, 1):
                para_sentences = split_paragraph_to_sentences(para)
                print_info(f"Paragraph {i}: {len(para)} chars, {len(para_sentences)} sentences")

        # Refine parameters for controlling pause/break at punctuation
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt=f"[break_{args.break_level}]",
            show_tqdm=not args.quiet,
        )

        # Audio generation parameters
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=spk,
            prompt=f"[speed_{args.speed}]",
            temperature=0.3,
            top_P=0.7,
            top_K=20,
            repetition_penalty=1.05,
            max_new_token=2048,
            ensure_non_empty=True,
        )

        # Process all sentences in batches
        sentence_audios = [None] * num_sentences
        num_batches = (num_sentences + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sentences)
            batch_sentences = all_sentences[start_idx:end_idx]

            if not args.quiet:
                print_info(f"Processing batch {batch_idx + 1}/{num_batches} (sentences {start_idx + 1}-{end_idx})...")

            # Batch inference
            wavs = chat.infer(
                batch_sentences,
                skip_refine_text=False,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
                use_decoder=True,
                do_text_normalization=False,
                do_homophone_replacement=False,
                split_text=False,
            )

            # Store results
            for i, wav in enumerate(wavs):
                sentence_idx = start_idx + i
                if wav is not None and len(wav) > 0:
                    normalized = float_to_int16(wav)
                    original = (wav * 32767).astype(np.int16)
                    audio_data = (normalized * 0.7 + original * 0.3).astype(np.int16)
                    sentence_audios[sentence_idx] = audio_data

        # Merge with proper pause durations
        # 0.5s between sentences within a paragraph, 1.0s between paragraphs
        if not args.quiet:
            print_info("Merging audio with pauses...")

        final_segments = []
        sample_rate = 24000
        sentence_pause = np.zeros(int(sample_rate * 0.5), dtype=np.int16)
        paragraph_pause = np.zeros(int(sample_rate * 1.0), dtype=np.int16)

        for para_idx in range(num_paragraphs):
            para_start = paragraph_boundaries[para_idx]
            para_end = paragraph_boundaries[para_idx + 1]

            # Add paragraph pause before (except for first paragraph)
            if para_idx > 0 and final_segments:
                final_segments.append(paragraph_pause)

            # Add sentences within this paragraph
            for sent_idx in range(para_start, para_end):
                if sentence_audios[sent_idx] is not None:
                    # Add sentence pause before (except for first sentence in paragraph)
                    if sent_idx > para_start and final_segments and not np.array_equal(final_segments[-1], paragraph_pause):
                        final_segments.append(sentence_pause)
                    final_segments.append(sentence_audios[sent_idx])

            if not args.quiet:
                para_duration = sum(len(sentence_audios[i]) for i in range(para_start, para_end) if sentence_audios[i] is not None) / sample_rate
                print_info(f"Paragraph {para_idx + 1} done: {para_duration:.2f}s")

        # Concatenate all segments
        if final_segments:
            merged_audio = np.concatenate(final_segments)
        else:
            merged_audio = np.array([], dtype=np.int16)

        # Final overall normalization
        if len(merged_audio) > 0:
            merged_audio_float = merged_audio.astype(np.float32) / 32767.0
            merged_audio = float_to_int16(merged_audio_float)

        wavfile.write(args.output_audio, 24000, merged_audio)
        duration = len(merged_audio) / 24000

        if not args.quiet:
            print_success(f"Audio saved: {args.output_audio}")
            print_info(f"Duration: {duration:.2f} seconds")

    generated_files = [args.output_audio]
    whisper_result = None

    # 6. Generate subtitles (if not skipped)
    if not args.skip_subtitles:
        if not args.quiet:
            print_step(2, total_steps, "Generating subtitles...")

        # Auto-derive SRT path if not specified
        output_srt = args.output_srt
        if output_srt is None:
            output_srt = args.output_audio.replace('.wav', '.srt')

        output_json = output_srt.replace('.srt', '.json') if not args.no_json else None

        whisper_result = generate_subtitles(
            audio_path=args.output_audio,
            language=args.language,
            whisper_model_size=args.whisper_model,
            output_srt_path=output_srt,
            output_json_path=output_json,
            generate_json=not args.no_json,
            quiet=args.quiet
        )

        generated_files.append(output_srt)
        if output_json:
            generated_files.append(output_json)

    # 7. Print summary
    if args.quiet:
        print_final_paths_only(generated_files)
    else:
        stats = {
            'duration': duration,
            'segments': len(whisper_result['segments']) if whisper_result else 0,
            'characters': len(text),
            'chars_per_sec': len(text) / duration if duration > 0 else 0,
            'files': generated_files
        }
        print_summary(stats)


# ========================================
# Argument Parser
# ========================================

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
  %(prog)s --input article.txt --speed 2
  %(prog)s --input story.txt --output-audio story.wav
  %(prog)s --text "Quick test" --skip-subtitles
  %(prog)s --input data.txt --quiet
        """
    )

    # Input group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Direct text input to convert to speech"
    )
    input_group.add_argument(
        "--input",
        type=str,
        help="Read text from file (UTF-8 encoded)"
    )

    # Output options
    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--output-srt",
        type=str,
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
        help="Disable text normalization (normalization is enabled by default to avoid invalid character warnings)"
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


# ========================================
# Entry Point
# ========================================

def main():
    """CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Validate speed
        validate_speed(args.speed)
        validate_language(args.language)

        # Run main workflow
        run_tts_with_subtitles(args)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()