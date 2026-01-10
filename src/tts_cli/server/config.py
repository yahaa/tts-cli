"""Server configuration."""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the TTS server."""

    host: str = "0.0.0.0"
    port: int = 8000
    mongodb_uri: str = "mongodb://localhost:27017"
    db_name: str = "tts-server"
    output_dir: str = "./tts_output"
    num_workers: int = 1
    cleanup_hours: int = 24
    max_text_length: int = 100000

    # TTS defaults
    default_language: str = "en"
    default_speed: int = 3
    default_break_level: int = 5
    default_max_length: int = 500
    default_whisper_model: str = "base"
