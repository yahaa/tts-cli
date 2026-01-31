"""Qwen3-TTS wrapper for tts-cli.

This module handles text-to-speech generation using Qwen3-TTS,
with GPU memory management and batch processing for optimal performance.

Includes backward-compatible wrappers for ChatTTS API to maintain existing functionality.
"""

from typing import List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from .audio import float_to_int16
from .utils import print_info, print_success
from .voice import load_voice_prompt, save_voice_prompt

# Model cache to avoid reloading
_MODEL_CACHE = {}


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


def estimate_memory_per_chunk_mb(chunk_chars: int = 1000, fp16: bool = True) -> float:
    """
    Estimate GPU memory usage per chunk for Qwen3-TTS.

    Args:
        chunk_chars: Number of characters per chunk
        fp16: Whether using FP16 (unused, kept for API compatibility)

    Returns:
        Estimated memory usage in MB per chunk
    """
    # Per-chunk inference memory (rough estimate for Qwen3-TTS)
    # ~300-500MB per chunk depending on length
    per_chunk_memory = 300 + (chunk_chars / 1000) * 200

    return per_chunk_memory


def calculate_optimal_batch_size(
    num_chunks: int,
    chunk_chars: int = 1000,
    reserved_memory_mb: float = 1024,
    min_batch: int = 1,
    max_batch: int = 4,
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    Args:
        num_chunks: Total number of chunks to process
        chunk_chars: Average characters per chunk
        reserved_memory_mb: Memory to reserve for system (default 1GB)
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size
    """
    free_memory = get_gpu_free_memory_mb()

    if free_memory <= 0:
        # No GPU or can't detect, use conservative batch size
        return min(min_batch, num_chunks)

    # Calculate available memory for batching
    available_memory = free_memory - reserved_memory_mb

    if available_memory <= 0:
        return min(min_batch, num_chunks)

    # Estimate memory per chunk
    memory_per_chunk = estimate_memory_per_chunk_mb(chunk_chars)

    # Calculate batch size
    if memory_per_chunk > 0:
        batch_size = int(available_memory / memory_per_chunk)
    else:
        batch_size = max_batch

    # Clamp to min/max and actual number of chunks
    batch_size = max(min_batch, min(batch_size, max_batch, num_chunks))

    return batch_size


def get_default_model_variant(mode: str) -> str:
    """
    Get default model variant for given voice mode.

    Args:
        mode: Voice mode ('custom', 'design', 'clone')

    Returns:
        Model variant string
    """
    if mode == "custom":
        return "1.7B-CustomVoice"
    elif mode == "design":
        return "1.7B-VoiceDesign"
    elif mode == "clone":
        return "1.7B-Base"
    return "1.7B-Base"


def init_qwen_tts(
    variant: str = "1.7B-Base",
    device: str = "auto",
    use_flash_attn: bool = True,
    quiet: bool = False,
):
    """
    Initialize and cache Qwen3-TTS model.

    Args:
        variant: Model variant to load
        device: Device to use ('auto', 'cuda', 'cpu')
        use_flash_attn: Whether to use FlashAttention 2 (reduces memory)
        quiet: Suppress initialization messages

    Returns:
        Qwen3-TTS model instance
    """
    # Check cache first
    cache_key = f"{variant}_{device}_{use_flash_attn}"
    if cache_key in _MODEL_CACHE:
        if not quiet:
            print_info(f"Using cached Qwen3-TTS model ({variant})")
        return _MODEL_CACHE[cache_key]

    if not quiet:
        print_info("Initializing Qwen3-TTS model...")
        print_info(f"Model variant: {variant}")
        print_info(f"Device: {device}")
        if use_flash_attn:
            print_info("FlashAttention 2: enabled (reduces GPU memory)")

    try:
        from qwen_tts import Qwen3TTSModel

        # Prepare initialization arguments
        model_name = f"Qwen/Qwen3-TTS-12Hz-{variant}"

        # Determine device_map - avoid "auto" which may cause meta device issues
        if device == "auto":
            # Explicitly choose device instead of using "auto"
            if torch.cuda.is_available():
                device_map = "cuda"
            else:
                device_map = "cpu"
        else:
            device_map = device

        # Prepare kwargs
        kwargs = {
            "device_map": device_map,
        }

        # Add FlashAttention 2 if requested
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        # Try to initialize model using from_pretrained
        try:
            model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
        except Exception as e:
            # If FlashAttention 2 failed, retry without it
            if use_flash_attn and "flash_attn" in str(e).lower():
                if not quiet:
                    print_info(
                        "FlashAttention 2 not available, falling back to standard attention"
                    )
                # Remove flash attention and retry
                kwargs.pop("attn_implementation", None)
                model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            else:
                raise

        # Cache the model
        _MODEL_CACHE[cache_key] = model

        if not quiet:
            print_success(f"Model loaded: {variant}")

        return model

    except ImportError as e:
        raise ImportError(
            f"Failed to import qwen_tts: {e}\n\n"
            "Install Qwen3-TTS with:\n"
            "  pip install qwen-tts\n\n"
            "For better performance with FlashAttention 2:\n"
            "  pip install flash-attn --no-build-isolation"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Qwen3-TTS model: {e}")


def generate_with_custom_voice(
    model,
    text: str,
    speaker: str = "Ryan",
    language: str = "auto",
    instruct: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate audio using custom voice (premium speakers).

    Args:
        model: Qwen3-TTS model instance
        text: Text to synthesize
        speaker: Speaker name
        language: Language code
        instruct: Optional style instruction

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Generate using Qwen3-TTS custom voice API
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=speaker, instruct=instruct
        )

        # Convert to numpy array if needed
        if isinstance(wavs, torch.Tensor):
            audio = wavs.cpu().numpy()
        else:
            audio = np.array(wavs)

        # Ensure single channel
        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Convert float to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = float_to_int16(audio)

        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Failed to generate audio with custom voice: {e}")


def generate_with_voice_design(
    model, text: str, voice_description: str, language: str = "auto"
) -> Tuple[np.ndarray, int]:
    """
    Generate audio using voice design (natural language description).

    Args:
        model: Qwen3-TTS model instance
        text: Text to synthesize
        voice_description: Natural language voice description
        language: Language code

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Generate using Qwen3-TTS voice design API
        wavs, sr = model.generate_voice_design(
            text=text, language=language, voice_description=voice_description
        )

        # Convert to numpy array
        if isinstance(wavs, torch.Tensor):
            audio = wavs.cpu().numpy()
        else:
            audio = np.array(wavs)

        # Ensure single channel
        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Convert float to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = float_to_int16(audio)

        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Failed to generate audio with voice design: {e}")


def generate_with_voice_clone(
    model, text: str, voice_prompt, language: str = "auto"
) -> Tuple[np.ndarray, int]:
    """
    Generate audio using voice clone prompt.

    Args:
        model: Qwen3-TTS model instance
        text: Text to synthesize
        voice_prompt: Voice clone prompt from create_voice_clone_prompt()
        language: Language code

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Generate using Qwen3-TTS voice clone API
        wavs, sr = model.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=voice_prompt
        )

        # Convert to numpy array
        if isinstance(wavs, torch.Tensor):
            audio = wavs.cpu().numpy()
        else:
            audio = np.array(wavs)

        # Ensure single channel
        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Convert float to int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = float_to_int16(audio)

        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Failed to generate audio with voice clone: {e}")


def generate_batch(
    model,
    texts: List[str],
    mode: str = "custom",
    speaker: Optional[str] = None,
    voice_description: Optional[str] = None,
    voice_prompt=None,
    language: str = "auto",
    instruct: Optional[str] = None,
) -> List[Tuple[np.ndarray, int]]:
    """
    Batch generation for multiple text chunks.

    Args:
        model: Qwen3-TTS model instance
        texts: List of text chunks to synthesize
        mode: Voice mode ('custom', 'design', 'clone')
        speaker: Speaker name for custom mode
        voice_description: Voice description for design mode
        voice_prompt: Voice prompt for clone mode
        language: Language code
        instruct: Optional style instruction

    Returns:
        List of (audio_array, sample_rate) tuples
    """
    results = []

    # TODO: Check if Qwen3-TTS supports true batch inference with list input
    # For now, process sequentially
    for text in texts:
        if mode == "custom":
            audio, sr = generate_with_custom_voice(
                model, text, speaker=speaker, language=language, instruct=instruct
            )
        elif mode == "design":
            audio, sr = generate_with_voice_design(
                model, text, voice_description=voice_description, language=language
            )
        elif mode == "clone":
            audio, sr = generate_with_voice_clone(
                model, text, voice_prompt=voice_prompt, language=language
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        results.append((audio, sr))

    return results


# ========================================
# Backward-Compatible Wrappers for ChatTTS API
# ========================================


def init_chat_tts(quiet: bool = False):
    """
    Legacy name for init_qwen_tts (backward compatibility).

    Args:
        quiet: Suppress progress messages

    Returns:
        Initialized Qwen3-TTS model instance
    """
    return init_qwen_tts(variant="1.7B-CustomVoice", quiet=quiet)


def sample_random_speaker(chat):
    """
    Legacy ChatTTS API - returns default preset speaker.

    Args:
        chat: Model instance (unused, kept for API compatibility)

    Returns:
        Default speaker name
    """
    return "Ryan"  # Default to Ryan (English male voice)


def load_speaker(speaker_file: str):
    """
    Load speaker/voice from file (auto-detect format).

    Args:
        speaker_file: Path to speaker/voice file

    Returns:
        Voice prompt for Qwen-TTS

    Raises:
        ValueError: If file is .pt (ChatTTS format, not supported)
    """
    if speaker_file.endswith(".pt"):
        raise ValueError(
            "ChatTTS speaker files (.pt) are not compatible with Qwen-TTS.\n\n"
            "To clone a voice:\n"
            "  tts-cli --mode clone --reference-audio <file.wav> \\\n"
            "          --reference-text '<transcript>' \\\n"
            "          --save-speaker voice.qwen-voice\n"
        )
    return load_voice_prompt(speaker_file)


def save_speaker(voice_prompt, speaker_file: str) -> None:
    """
    Save voice prompt to file.

    Args:
        voice_prompt: Voice prompt to save
        speaker_file: Output file path
    """
    if not speaker_file.endswith(".qwen-voice"):
        speaker_file += ".qwen-voice"
    save_voice_prompt(voice_prompt, speaker_file)


def generate_audio(
    chat,
    text: str,
    speed: int,
    output_path: str,
    speaker_file: Optional[str] = None,
    speaker_name: Optional[str] = None,
    save_speaker_file: Optional[str] = None,
    break_level: int = 5,  # noqa: ARG001 - kept for API compatibility
    quiet: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Generate audio - backward compatible API.

    Args:
        chat: Qwen3-TTS model instance
        text: Input text to convert to speech
        speed: Speech speed (0-9)
        output_path: Output audio file path
        speaker_file: Optional speaker/voice file to load (.qwen-voice)
        speaker_name: Optional preset speaker name (e.g. Ryan, Vivian)
        save_speaker_file: Optional file path to save voice
        break_level: Pause strength (unused, kept for compatibility)
        quiet: Suppress progress messages

    Returns:
        Tuple of (audio_data, duration_seconds)
    """
    # Map speed to instruct
    instruct = None
    if speed <= 2:
        instruct = "speak slowly and clearly"
    elif speed >= 7:
        instruct = "speak quickly"

    if not quiet:
        if instruct:
            print_info(f"Style instruction: {instruct}")

    # Generate based on whether speaker file provided
    if speaker_file:
        if not quiet:
            print_info(f"Loading voice from: {speaker_file}")
        voice_prompt = load_speaker(speaker_file)
        audio, sr = generate_with_voice_clone(chat, text, voice_prompt)

        # Save voice if requested
        if save_speaker_file:
            save_speaker(voice_prompt, save_speaker_file)
            if not quiet:
                print_info(f"Voice saved to: {save_speaker_file}")
    else:
        preset = speaker_name or "Ryan"
        if not quiet:
            print_info(f"Using preset speaker: {preset}")
        audio, sr = generate_with_custom_voice(
            chat, text, speaker=preset, instruct=instruct
        )

    # Save audio
    wavfile.write(output_path, sr, audio)

    duration = len(audio) / sr

    if not quiet:
        print_success(f"Audio saved: {output_path}")
        print_info(f"Duration: {duration:.2f} seconds")

    return audio, duration


def generate_audio_batch(
    chat,
    texts: List[str],
    speed: int,
    spk,
    break_level: int = 5,  # noqa: ARG001 - kept for API compatibility
    quiet: bool = False,
) -> List[np.ndarray]:
    """
    Batch generation - backward compatible API.

    Args:
        chat: Qwen3-TTS model instance
        texts: List of text chunks to convert
        speed: Speech speed (0-9)
        spk: Speaker (string for preset name, or voice prompt object)
        break_level: Pause strength (unused, kept for compatibility)
        quiet: Suppress progress messages

    Returns:
        List of audio arrays (int16)
    """
    # Map speed to instruct
    instruct = None
    if speed <= 2:
        instruct = "speak slowly"
    elif speed >= 7:
        instruct = "speak quickly"

    # Determine mode based on spk type
    if isinstance(spk, str):
        mode = "custom"
        speaker = spk
        voice_prompt = None
        if not quiet:
            print_info(f"Batch generation with preset speaker: {speaker}")
    else:
        mode = "clone"
        speaker = None
        voice_prompt = spk
        if not quiet:
            print_info("Batch generation with cloned voice")

    results = generate_batch(
        chat,
        texts,
        mode=mode,
        speaker=speaker,
        voice_prompt=voice_prompt,
        instruct=instruct,
    )

    # Extract just audio arrays
    return [audio for audio, sr in results]
