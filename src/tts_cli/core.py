"""Main workflow orchestrator for tts-cli."""

from typing import Optional
import numpy as np
import scipy.io.wavfile as wavfile

from .audio import (
    float_to_int16, merge_with_pauses, normalize_audio,
    get_audio_duration, SAMPLE_RATE
)
from .text_processor import (
    split_text_intelligently, split_paragraph_to_sentences,
    normalize_text_for_tts
)
from .tts import (
    init_chat_tts, load_speaker, save_speaker, sample_random_speaker,
    generate_audio, generate_audio_batch,
    get_gpu_free_memory_mb, estimate_memory_per_chunk_mb, calculate_optimal_batch_size
)
from .subtitle import generate_subtitles
from .utils import (
    read_text_from_file, check_dependencies,
    print_header, print_step, print_info, print_success,
    print_summary, print_final_paths_only
)


class TTSConfig:
    """Configuration for TTS generation."""

    def __init__(
        self,
        text: Optional[str] = None,
        input_file: Optional[str] = None,
        output_audio: str = "output.wav",
        output_srt: Optional[str] = None,
        speed: int = 3,
        language: str = "en",
        break_level: int = 5,
        speaker: Optional[str] = None,
        save_speaker: Optional[str] = None,
        max_length: Optional[int] = None,
        no_normalize: bool = False,
        whisper_model: str = "base",
        skip_subtitles: bool = False,
        no_json: bool = True,
        quiet: bool = False,
    ):
        self.text = text
        self.input_file = input_file
        self.output_audio = output_audio
        self.output_srt = output_srt
        self.speed = speed
        self.language = language
        self.break_level = break_level
        self.speaker = speaker
        self.save_speaker = save_speaker
        self.max_length = max_length
        self.no_normalize = no_normalize
        self.whisper_model = whisper_model
        self.skip_subtitles = skip_subtitles
        self.no_json = no_json
        self.quiet = quiet


def run_tts_with_subtitles(config: TTSConfig) -> None:
    """
    Main workflow orchestrator.

    Args:
        config: TTS configuration
    """
    # 1. Validate inputs
    if config.text and config.input_file:
        raise ValueError("Cannot specify both text and input file")
    if not config.text and not config.input_file:
        raise ValueError("Must specify either text or input file")

    # 2. Check dependencies
    require_whisper = not config.skip_subtitles
    check_dependencies(require_whisper)

    # 3. Read text input
    if config.input_file:
        text = read_text_from_file(config.input_file)
    else:
        text = config.text

    # Validate text is not empty
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    # Normalize text by default (unless disabled)
    if not config.no_normalize:
        original_length = len(text)
        text = normalize_text_for_tts(text)
        if not config.quiet:
            print_info(f"Text normalized ({original_length} -> {len(text)} chars)")

    # 4. Print header (if not quiet)
    if not config.quiet:
        print_header()
        print_info(f"Input text length: {len(text)} characters")

    # 5. Initialize ChatTTS
    chat = init_chat_tts(quiet=config.quiet)

    # 6. Check if text needs splitting
    text_chunks = []
    if config.max_length and len(text) > config.max_length:
        text_chunks = split_text_intelligently(text, config.max_length)
        if not config.quiet:
            print_info(f"Text split into {len(text_chunks)} chunks for better quality")
    else:
        text_chunks = [text]

    # 7. Generate audio
    total_steps = 2 if not config.skip_subtitles else 1

    if len(text_chunks) == 1:
        # Single chunk - direct generation
        if not config.quiet:
            print_step(1, total_steps, "Generating audio...")

        _, duration = generate_audio(
            chat=chat,
            text=text_chunks[0],
            speed=config.speed,
            output_path=config.output_audio,
            speaker_file=config.speaker,
            save_speaker_file=config.save_speaker,
            break_level=config.break_level,
            quiet=config.quiet
        )
    else:
        # Multiple chunks - batch processing with GPU parallelism
        duration = _generate_audio_multi_chunk(
            chat=chat,
            text_chunks=text_chunks,
            config=config,
            total_steps=total_steps
        )

    generated_files = [config.output_audio]
    whisper_result = None

    # 8. Generate subtitles (if not skipped)
    if not config.skip_subtitles:
        if not config.quiet:
            print_step(2, total_steps, "Generating subtitles...")

        # Auto-derive SRT path if not specified
        output_srt = config.output_srt
        if output_srt is None:
            output_srt = config.output_audio.replace('.wav', '.srt')

        output_json = output_srt.replace('.srt', '.json') if not config.no_json else None

        whisper_result = generate_subtitles(
            audio_path=config.output_audio,
            language=config.language,
            whisper_model_size=config.whisper_model,
            output_srt_path=output_srt,
            output_json_path=output_json,
            generate_json=not config.no_json,
            quiet=config.quiet
        )

        generated_files.append(output_srt)
        if output_json:
            generated_files.append(output_json)

    # 9. Print summary
    if config.quiet:
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


def _generate_audio_multi_chunk(
    chat,
    text_chunks: list,
    config: TTSConfig,
    total_steps: int
) -> float:
    """
    Generate audio for multiple text chunks with batch processing.

    Args:
        chat: ChatTTS instance
        text_chunks: List of text chunks (paragraphs)
        config: TTS configuration
        total_steps: Total number of steps for progress display

    Returns:
        Total audio duration in seconds
    """
    num_paragraphs = len(text_chunks)

    # Split each paragraph into sentences
    # Filter out sentences that are too short (< 5 chars) - they cause ChatTTS errors
    MIN_SENTENCE_LENGTH = 5
    all_sentences = []
    paragraph_boundaries = [0]

    for para in text_chunks:
        sentences = split_paragraph_to_sentences(para)
        if not sentences:
            sentences = [para]
        # Filter short sentences, keep only valid ones
        valid_sentences = [s for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]
        if not valid_sentences and para.strip():
            # If all sentences filtered out but paragraph has content, use it as-is
            valid_sentences = [para.strip()] if len(para.strip()) >= MIN_SENTENCE_LENGTH else []
        all_sentences.extend(valid_sentences)
        paragraph_boundaries.append(len(all_sentences))

    num_sentences = len(all_sentences)
    if num_sentences == 0:
        raise ValueError("No valid text content found. Text may be too short or contain only special characters.")
    avg_sentence_chars = sum(len(s) for s in all_sentences) // max(num_sentences, 1)

    # Calculate optimal batch size based on GPU memory
    batch_size = calculate_optimal_batch_size(
        num_chunks=num_sentences,
        chunk_chars=avg_sentence_chars
    )

    if not config.quiet:
        free_mem = get_gpu_free_memory_mb()
        mem_per_chunk = estimate_memory_per_chunk_mb(avg_sentence_chars)
        print_step(1, total_steps, f"Generating audio ({num_paragraphs} paragraphs, {num_sentences} sentences)...")
        print_info(f"GPU free memory: {free_mem:.0f} MB")
        print_info(f"Estimated memory per sentence: {mem_per_chunk:.0f} MB")
        print_info(f"Auto batch size: {batch_size}")

    # Load or create speaker (ensure consistency across chunks)
    if config.speaker:
        spk = load_speaker(config.speaker)
    else:
        spk = sample_random_speaker(chat)
        if config.save_speaker:
            save_speaker(spk, config.save_speaker)
            if not config.quiet:
                print_info(f"Speaker saved to: {config.save_speaker}")

    if not config.quiet:
        for i, para in enumerate(text_chunks, 1):
            para_sentences = split_paragraph_to_sentences(para)
            print_info(f"Paragraph {i}: {len(para)} chars, {len(para_sentences)} sentences")

    # Process all sentences in batches
    sentence_audios = [None] * num_sentences
    num_batches = (num_sentences + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_sentences)
        batch_sentences = all_sentences[start_idx:end_idx]

        if not config.quiet:
            print_info(f"Processing batch {batch_idx + 1}/{num_batches} (sentences {start_idx + 1}-{end_idx})...")

        # Batch inference
        batch_audios = generate_audio_batch(
            chat=chat,
            texts=batch_sentences,
            speed=config.speed,
            spk=spk,
            break_level=config.break_level,
            quiet=config.quiet
        )

        # Store results
        for i, audio in enumerate(batch_audios):
            sentence_audios[start_idx + i] = audio

    # Merge with proper pause durations
    if not config.quiet:
        print_info("Merging audio with pauses...")

    merged_audio = merge_with_pauses(
        audio_segments=sentence_audios,
        paragraph_boundaries=paragraph_boundaries,
        sample_rate=SAMPLE_RATE,
        sentence_pause=0.5,
        paragraph_pause=1.0
    )

    # Final normalization
    if len(merged_audio) > 0:
        merged_audio = normalize_audio(merged_audio)

    # Save audio
    wavfile.write(config.output_audio, SAMPLE_RATE, merged_audio)
    duration = get_audio_duration(merged_audio)

    if not config.quiet:
        print_success(f"Audio saved: {config.output_audio}")
        print_info(f"Duration: {duration:.2f} seconds")

    return duration
