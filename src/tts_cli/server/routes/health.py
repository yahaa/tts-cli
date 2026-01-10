"""Health check routes."""

from fastapi import APIRouter

from ... import __version__
from ..db import Database
from ..models import HealthResponse
from ..services.tts_engine import TTSEngine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    tts_engine = TTSEngine()

    return HealthResponse(
        status="healthy",
        model_loaded=tts_engine.is_loaded(),
        version=__version__,
        mongodb_connected=Database.is_connected(),
    )
