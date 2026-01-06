"""ChatTTS wrapper for tts-cli.

This module handles text-to-speech generation using ChatTTS,
with GPU memory management and batch processing for optimal performance.
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import scipy.io.wavfile as wavfile

from .audio import float_to_int16, SAMPLE_RATE
from .utils import print_info, print_success


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
        free_memory = torch.cuda.mem_get_info()[0]
        return free_memory / (1024 * 1024)
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
# ChatTTS Initialization
# ========================================

def init_chat_tts(quiet: bool = False):
    """
    Initialize ChatTTS model.

    Args:
        quiet: Suppress progress messages

    Returns:
        Initialized ChatTTS.Chat instance
    """
    import ChatTTS

    if not quiet:
        print_info("Loading ChatTTS model...")

    chat = ChatTTS.Chat()

    # Try to load from huggingface cache first
    hf_cache_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub/models--2Noise--ChatTTS/snapshots")
    )

    loaded = False

    if os.path.exists(hf_cache_path):
        snapshots = [d for d in os.listdir(hf_cache_path)
                    if os.path.isdir(os.path.join(hf_cache_path, d))]
        if snapshots:
            custom_path = os.path.join(hf_cache_path, snapshots[0])
            if not quiet:
                print_info(f"Trying to load from cache: {custom_path}")
            try:
                loaded = chat.load(source="custom", custom_path=custom_path, compile=False)
            except Exception as e:
                if not quiet:
                    print_info(f"Custom load failed: {e}, falling back to huggingface")
                loaded = False

    # Fallback to huggingface if custom load failed
    if not loaded:
        if not quiet:
            print_info("Loading from huggingface...")
        loaded = chat.load(source="huggingface", compile=False)

    if not loaded:
        raise RuntimeError("Failed to load ChatTTS model")

    return chat


# ========================================
# Speaker Management
# ========================================

def load_speaker(speaker_file: str) -> torch.Tensor:
    """
    Load speaker embedding from file.

    Args:
        speaker_file: Path to speaker embedding file (.pt)

    Returns:
        Speaker embedding tensor
    """
    return torch.load(speaker_file, weights_only=True)


def save_speaker(spk: torch.Tensor, speaker_file: str) -> None:
    """
    Save speaker embedding to file.

    Args:
        spk: Speaker embedding tensor
        speaker_file: Path to save speaker embedding
    """
    torch.save(spk, speaker_file)


def sample_random_speaker(chat) -> torch.Tensor:
    """
    Sample a random speaker from ChatTTS.

    Args:
        chat: ChatTTS.Chat instance

    Returns:
        Speaker embedding tensor
    """
    return chat.sample_random_speaker()


# ========================================
# Audio Generation
# ========================================

def generate_audio(
    chat,
    text: str,
    speed: int,
    output_path: str,
    speaker_file: Optional[str] = None,
    save_speaker_file: Optional[str] = None,
    break_level: int = 5,
    quiet: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Generate audio using ChatTTS for a single text chunk.

    Args:
        chat: Initialized ChatTTS Chat instance
        text: Input text to convert to speech
        speed: Speech speed (0-9)
        output_path: Output audio file path
        speaker_file: Optional speaker embedding file to load
        save_speaker_file: Optional file path to save speaker embedding
        break_level: Pause strength at punctuation (0-7, default: 5)
        quiet: Suppress progress messages

    Returns:
        Tuple of (audio_data, duration_seconds)
    """
    import ChatTTS as CT

    # Validate text is not empty or too short
    if not text or len(text.strip()) < 2:
        raise ValueError("Text is too short to generate audio (minimum 2 characters)")

    if not quiet:
        print_info(f"Speed parameter: speed_{speed}")

    # Load or sample speaker embedding
    if speaker_file:
        if not quiet:
            print_info(f"Loading speaker from: {speaker_file}")
        spk = load_speaker(speaker_file)
    else:
        if not quiet:
            print_info("Sampling random speaker...")
        spk = sample_random_speaker(chat)

    # Save speaker if requested
    if save_speaker_file:
        save_speaker(spk, save_speaker_file)
        if not quiet:
            print_info(f"Speaker saved to: {save_speaker_file}")

    if not quiet:
        print_info("Generating speech...")

    # Refine parameters for controlling pause/break at punctuation
    params_refine_text = CT.Chat.RefineTextParams(
        prompt=f"[break_{break_level}]",
        show_tqdm=not quiet,
    )

    # Generate audio with skip_refine_text=True to avoid "narrow(): length must be non-negative" errors
    params_infer_code = CT.Chat.InferCodeParams(
        spk_emb=spk,
        prompt=f"[speed_{speed}]",
        temperature=0.3,
        top_P=0.7,
        top_K=20,
        repetition_penalty=1.05,
        max_new_token=2048,
    )

    wavs = chat.infer(
        [text],
        skip_refine_text=True,
        params_infer_code=params_infer_code,
        use_decoder=True,
        do_text_normalization=False,
        do_homophone_replacement=False,
    )

    # Save audio with adaptive normalization
    audio_data = float_to_int16(wavs[0])
    wavfile.write(output_path, SAMPLE_RATE, audio_data)

    duration = len(wavs[0]) / SAMPLE_RATE

    if not quiet:
        print_success(f"Audio saved: {output_path}")
        print_info(f"Duration: {duration:.2f} seconds")

    return audio_data, duration


def generate_audio_batch(
    chat,
    texts: List[str],
    speed: int,
    spk: torch.Tensor,
    break_level: int = 5,
    quiet: bool = False
) -> List[np.ndarray]:
    """
    Generate audio for multiple text chunks in a batch.

    This leverages GPU parallelism for faster generation.

    Args:
        chat: Initialized ChatTTS Chat instance
        texts: List of text chunks to convert
        speed: Speech speed (0-9)
        spk: Speaker embedding tensor
        break_level: Pause strength at punctuation (0-7)
        quiet: Suppress progress messages

    Returns:
        List of audio arrays (int16)
    """
    import ChatTTS as CT

    # Filter out empty or too-short texts to avoid ChatTTS errors
    # Minimum 5 chars needed for reliable audio generation
    MIN_TEXT_LENGTH = 5
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text and len(text.strip()) >= MIN_TEXT_LENGTH:
            valid_indices.append(i)
            valid_texts.append(text.strip())

    # If no valid texts, return list of None
    if not valid_texts:
        return [None] * len(texts)

    # Audio generation parameters
    # Note: ensure_non_empty=False to avoid "unexpected end at index" errors on Windows
    params_infer_code = CT.Chat.InferCodeParams(
        spk_emb=spk,
        prompt=f"[speed_{speed}]",
        temperature=0.3,
        top_P=0.7,
        top_K=20,
        repetition_penalty=1.05,
        max_new_token=2048,
        ensure_non_empty=False,
    )

    # Build result array with None for all positions
    audio_arrays = [None] * len(texts)

    # Try batch inference first
    wavs = None
    batch_success = False
    try:
        wavs = chat.infer(
            valid_texts,
            skip_refine_text=True,
            params_infer_code=params_infer_code,
            use_decoder=True,
            do_text_normalization=False,
            do_homophone_replacement=False,
        )
        # Check if batch returned valid results for all texts
        if wavs is not None and len(wavs) == len(valid_texts):
            batch_success = True
    except Exception as e:
        if not quiet:
            print_info(f"Batch inference failed ({e}), falling back to sequential...")

    # If batch succeeded, process results
    if batch_success:
        for idx, wav in zip(valid_indices, wavs):
            if wav is not None and len(wav) > 0:
                audio_arrays[idx] = _convert_wav_to_int16(wav)
        return audio_arrays

    # Fallback: process one by one, skipping failures silently
    if not quiet:
        print_info("Processing sentences one by one...")

    for idx, text in zip(valid_indices, valid_texts):
        try:
            result = chat.infer(
                [text],
                skip_refine_text=True,
                params_infer_code=params_infer_code,
                use_decoder=True,
                do_text_normalization=False,
                do_homophone_replacement=False,
            )
            if result and len(result) > 0 and result[0] is not None and len(result[0]) > 0:
                audio_arrays[idx] = _convert_wav_to_int16(result[0])
        except Exception:
            # Skip failed sentences silently
            pass

    return audio_arrays


def _convert_wav_to_int16(wav: np.ndarray) -> np.ndarray:
    """Convert float wav to int16 with blended normalization."""
    normalized = float_to_int16(wav)
    original = (wav * 32767).astype(np.int16)
    # Blend normalized and original for balanced audio
    return (normalized * 0.7 + original * 0.3).astype(np.int16)
