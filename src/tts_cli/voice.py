"""Voice management for tts-cli."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch

# Supported speakers for custom voice mode
SUPPORTED_SPEAKERS = [
    "Vivian",  # Chinese Female
    "Serena",  # Chinese Female
    "Uncle_Fu",  # Chinese Male
    "Dylan",  # Chinese (Beijing) Male
    "Eric",  # Chinese (Sichuan) Male
    "Ryan",  # English Male
    "Aiden",  # English Male
    "Ono_Anna",  # Japanese Female
    "Sohee",  # Korean Female
]

VOICE_PROMPT_VERSION = "1.0"


def validate_speaker(speaker: str) -> bool:
    """
    Validate speaker name against supported speakers.

    Args:
        speaker: Speaker name to validate

    Returns:
        True if valid

    Raises:
        ValueError: If speaker name is not supported
    """
    if speaker not in SUPPORTED_SPEAKERS:
        raise ValueError(
            f"Invalid speaker '{speaker}'. Supported speakers: {', '.join(SUPPORTED_SPEAKERS)}"
        )
    return True


def validate_reference_audio(audio_path: str) -> Tuple[bool, str]:
    """
    Validate reference audio file meets requirements for voice cloning.

    Args:
        audio_path: Path to reference audio file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(audio_path):
        return False, f"Reference audio file not found: {audio_path}"

    if not os.path.isfile(audio_path):
        return False, f"Reference audio path is not a file: {audio_path}"

    # Check file extension
    valid_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    ext = Path(audio_path).suffix.lower()
    if ext not in valid_extensions:
        return (
            False,
            f"Unsupported audio format '{ext}'. Supported: {', '.join(valid_extensions)}",
        )

    # TODO: Add duration check (>= 3 seconds) when we implement audio loading
    # For now, just check file exists and has valid extension

    return True, ""


def create_voice_clone_prompt(
    model,
    reference_audio: str,
    reference_text: str,
):
    """
    Create reusable voice clone prompt from reference audio.

    Args:
        model: Qwen3-TTS model instance
        reference_audio: Path to reference audio file (3+ seconds)
        reference_text: Transcript of the reference audio

    Returns:
        Voice clone prompt (tensor/dict)

    Raises:
        ValueError: If reference audio is invalid
    """
    # Validate reference audio
    is_valid, error_msg = validate_reference_audio(reference_audio)
    if not is_valid:
        raise ValueError(error_msg)

    # Validate reference text
    if not reference_text or not reference_text.strip():
        raise ValueError("Reference text cannot be empty")

    # Create voice clone prompt using Qwen3-TTS
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=reference_audio, ref_text=reference_text
    )

    return voice_prompt


def save_voice_prompt(
    voice_prompt,
    output_path: str,
    reference_text: str = "",
    language: str = "auto",
    model_variant: str = "1.7B-Base",
) -> None:
    """
    Save voice clone prompt to file with metadata.

    Args:
        voice_prompt: Voice prompt from create_voice_clone_prompt()
        output_path: Path to save .qwen-voice file
        reference_text: Original reference text
        language: Language of the reference
        model_variant: Model variant used
    """
    # Ensure .qwen-voice extension
    if not output_path.endswith(".qwen-voice"):
        output_path = output_path + ".qwen-voice"

    # Create metadata
    metadata = {
        "version": VOICE_PROMPT_VERSION,
        "prompt": voice_prompt,
        "reference_text": reference_text,
        "language": language,
        "created_at": datetime.now().isoformat(),
        "model_variant": model_variant,
    }

    # Save as PyTorch file
    torch.save(metadata, output_path)


def load_voice_prompt(voice_file: str):
    """
    Load saved voice clone prompt from file.

    Args:
        voice_file: Path to .qwen-voice file (extension optional)

    Returns:
        Voice prompt (tensor/dict)

    Raises:
        FileNotFoundError: If voice file doesn't exist
        ValueError: If voice file is invalid or version mismatch
    """
    # Auto-add .qwen-voice extension if not present
    if not voice_file.endswith(".qwen-voice"):
        voice_file = voice_file + ".qwen-voice"

    if not os.path.exists(voice_file):
        raise FileNotFoundError(f"Voice prompt file not found: {voice_file}")

    try:
        # PyTorch 2.6+ requires weights_only=False for custom classes
        metadata = torch.load(voice_file, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load voice prompt file: {e}")

    # Validate version
    if "version" not in metadata:
        raise ValueError("Invalid voice prompt file: missing version")

    if metadata["version"] != VOICE_PROMPT_VERSION:
        raise ValueError(
            f"Voice prompt version mismatch: expected {VOICE_PROMPT_VERSION}, "
            f"got {metadata['version']}"
        )

    # Return the prompt
    return metadata.get("prompt")


def validate_voice_config(
    mode: str,
    speaker: Optional[str] = None,
    voice_description: Optional[str] = None,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    voice_prompt_file: Optional[str] = None,
) -> None:
    """
    Validate voice configuration based on mode.

    Args:
        mode: Voice mode ('custom', 'design', 'clone')
        speaker: Speaker name for custom mode
        voice_description: Voice description for design mode
        reference_audio: Reference audio for clone mode
        reference_text: Reference text for clone mode
        voice_prompt_file: Saved voice prompt for clone mode

    Raises:
        ValueError: If configuration is invalid for the mode
    """
    if mode == "custom":
        if not speaker:
            raise ValueError("Custom voice mode requires --speaker")
        validate_speaker(speaker)

    elif mode == "design":
        if not voice_description:
            raise ValueError("Voice design mode requires --voice-description")

    elif mode == "clone":
        # Either provide reference audio + text, OR load saved voice prompt
        has_reference = reference_audio and reference_text
        has_saved_prompt = voice_prompt_file

        if not has_reference and not has_saved_prompt:
            raise ValueError(
                "Voice clone mode requires either:\n"
                "  1. --reference-audio + --reference-text (to create new clone), OR\n"
                "  2. --voice-prompt (to load saved clone)"
            )

        if has_reference:
            is_valid, error_msg = validate_reference_audio(reference_audio)
            if not is_valid:
                raise ValueError(error_msg)

    else:
        raise ValueError(
            f"Invalid voice mode '{mode}'. Must be 'custom', 'design', or 'clone'"
        )
