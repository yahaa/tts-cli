"""Pydantic request/response models for the TTS API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CreateTtsTaskRequest(BaseModel):
    """Request model for creating a TTS task."""

    text: str = Field(..., description="Text to convert to speech", max_length=100000)
    language: str = Field(default="en", description="Language code: 'en' or 'zh'")
    speed: int = Field(default=3, ge=0, le=9, description="Speech speed 0-9")
    break_level: int = Field(
        default=5, ge=0, le=7, description="Punctuation pause strength 0-7"
    )
    speaker_id: Optional[str] = Field(
        default=None, description="Local speaker embedding file path (.pt)"
    )
    max_length: int = Field(
        default=500, ge=100, le=2000, description="Auto-split text threshold"
    )
    max_batch: int = Field(
        default=1, ge=1, le=10, description="Max chunks to process in parallel"
    )
    skip_subtitles: bool = Field(default=False, description="Skip subtitle generation")
    whisper_model: str = Field(
        default="base",
        description="Whisper model size: tiny/base/small/medium/large",
    )
    callback_url: Optional[str] = Field(
        default=None, description="URL for result callback (POST)"
    )


class CreateTtsTaskResponse(BaseModel):
    """Response model for creating a TTS task."""

    request_id: str
    task_id: str
    status: str


class TaskInfo(BaseModel):
    """Task information model."""

    task_id: str
    status: Literal["waiting", "processing", "success", "failed"]
    audio_url: Optional[str] = None
    subtitle_url: Optional[str] = None
    duration: Optional[float] = None
    character_count: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    create_time: datetime
    start_time: Optional[datetime] = None
    finish_time: Optional[datetime] = None


class DescribeTtsTaskResponse(BaseModel):
    """Response model for describing a TTS task."""

    request_id: str
    task: Optional[TaskInfo] = None
    error: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    version: str
    mongodb_connected: bool


class ErrorResponse(BaseModel):
    """Error response model."""

    request_id: str
    error: dict
