"""TTS engine singleton wrapper."""

import logging
import os
import threading
from typing import Optional

from ...core import TTSConfig, run_tts_with_subtitles
from ...tts import init_chat_tts

logger = logging.getLogger(__name__)


class TTSEngine:
    """Singleton wrapper for ChatTTS model."""

    _instance: Optional["TTSEngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._chat = None
        self._model_lock = threading.Lock()
        self._initialized = True

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._chat is not None

    def ensure_loaded(self) -> None:
        """Lazy load ChatTTS model."""
        if self._chat is None:
            with self._model_lock:
                if self._chat is None:
                    logger.info("Loading ChatTTS model...")
                    self._chat = init_chat_tts(quiet=True)
                    logger.info("ChatTTS model loaded")

    def generate(
        self,
        text: str,
        output_dir: str,
        task_id: str,
        language: str = "en",
        speed: int = 3,
        break_level: int = 5,
        speaker_id: Optional[str] = None,
        max_length: int = 500,
        max_batch: int = 1,
        skip_subtitles: bool = False,
        whisper_model: str = "base",
    ) -> tuple[str, Optional[str], float]:
        """
        Generate audio and optionally subtitles.

        Returns:
            Tuple of (audio_path, subtitle_path, duration)
        """
        self.ensure_loaded()

        # Create task output directory
        task_dir = os.path.join(output_dir, "tasks", task_id)
        os.makedirs(task_dir, exist_ok=True)

        audio_path = os.path.join(task_dir, "output.wav")
        subtitle_path = (
            os.path.join(task_dir, "output.srt") if not skip_subtitles else None
        )

        # Create config
        config = TTSConfig(
            text=text,
            output_audio=audio_path,
            output_srt=subtitle_path,
            speed=speed,
            language=language,
            break_level=break_level,
            speaker=speaker_id,
            max_length=max_length,
            max_batch=max_batch,
            skip_subtitles=skip_subtitles,
            whisper_model=whisper_model,
            quiet=True,
            no_json=True,
        )

        # Run TTS generation
        run_tts_with_subtitles(config)

        # Calculate duration from audio file
        import scipy.io.wavfile as wavfile

        sample_rate, audio_data = wavfile.read(audio_path)
        duration = len(audio_data) / sample_rate

        return audio_path, subtitle_path, duration
