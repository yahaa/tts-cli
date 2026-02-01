"""Audio processing utilities for tts-cli."""

import math
from typing import List

import numpy as np

# Try to import numba for JIT optimization
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy decorator if numba is not available
    def jit(nopython=True):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


# Default sample rate. Qwen3-TTS typically outputs 24kHz audio.
# If the model returns a different sample rate, callers should use that value
# instead of this constant (e.g. generate_audio returns sr from the model).
SAMPLE_RATE = 24000


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
    max_val = float(np.abs(audio).max())
    if max_val < 1e-10:
        # Silent audio — return zeros directly to avoid division by zero
        return np.zeros(len(audio), dtype=np.int16)
    am = int(math.ceil(max_val) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def merge_audio_files(
    audio_arrays: List[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    pause_duration: float = 0.7,
) -> np.ndarray:
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


def merge_with_pauses(
    audio_segments: List[np.ndarray],
    paragraph_boundaries: List[int],
    sample_rate: int = SAMPLE_RATE,
    sentence_pause: float = 0.5,
    paragraph_pause: float = 1.0,
) -> np.ndarray:
    """
    Merge audio segments with different pause durations for sentences and paragraphs.

    Args:
        audio_segments: List of audio arrays for each sentence
        paragraph_boundaries: List of indices where each paragraph starts
        sample_rate: Sample rate of the audio
        sentence_pause: Duration of silence between sentences (default: 0.5s)
        paragraph_pause: Duration of silence between paragraphs (default: 1.0s)

    Returns:
        Merged audio array
    """
    if not audio_segments:
        return np.array([], dtype=np.int16)

    num_paragraphs = len(paragraph_boundaries) - 1
    sentence_silence = np.zeros(int(sample_rate * sentence_pause), dtype=np.int16)
    paragraph_silence = np.zeros(int(sample_rate * paragraph_pause), dtype=np.int16)

    final_segments = []

    for para_idx in range(num_paragraphs):
        para_start = paragraph_boundaries[para_idx]
        para_end = paragraph_boundaries[para_idx + 1]

        # Add paragraph pause before (except for first paragraph)
        if para_idx > 0 and final_segments:
            final_segments.append(paragraph_silence)

        # Add sentences within this paragraph
        for sent_idx in range(para_start, para_end):
            if (
                audio_segments[sent_idx] is not None
                and len(audio_segments[sent_idx]) > 0
            ):
                # Add sentence pause before (except for first sentence in paragraph)
                if (
                    sent_idx > para_start
                    and final_segments
                    and not np.array_equal(final_segments[-1], paragraph_silence)
                ):
                    final_segments.append(sentence_silence)
                final_segments.append(audio_segments[sent_idx])

    # Concatenate all segments
    if final_segments:
        return np.concatenate(final_segments)
    return np.array([], dtype=np.int16)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize int16 audio to maximize dynamic range.

    Re-scales audio so the peak sample uses the full int16 range.
    This is intentionally a round-trip (int16 → float → int16) to
    re-maximize dynamic range after operations like concatenation.

    Args:
        audio: Audio data as int16 array

    Returns:
        Normalized audio as int16 array
    """
    if len(audio) == 0:
        return audio

    # Guard against silent audio
    if np.abs(audio).max() == 0:
        return audio

    # Convert to float, normalize, convert back
    audio_float = audio.astype(np.float32) / 32767.0
    return float_to_int16(audio_float)


def get_audio_duration(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Calculate audio duration in seconds.

    Args:
        audio: Audio data array
        sample_rate: Sample rate of the audio

    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate
