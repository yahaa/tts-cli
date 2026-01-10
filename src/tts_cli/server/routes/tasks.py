"""Task API routes."""

import os
import uuid

from fastapi import APIRouter, HTTPException, Query

from ..models import (
    CreateTtsTaskRequest,
    CreateTtsTaskResponse,
    DescribeTtsTaskResponse,
    TaskInfo,
)
from ..services.task_manager import TaskManager

router = APIRouter(tags=["tasks"])

# Will be set by app.py during startup
task_manager: TaskManager = None


def set_task_manager(manager: TaskManager) -> None:
    """Set the task manager instance."""
    global task_manager
    task_manager = manager


@router.post("/create_tts_task", response_model=CreateTtsTaskResponse)
async def create_tts_task(request: CreateTtsTaskRequest):
    """Create a new TTS task."""
    if task_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    # Validate speaker file if provided
    if request.speaker_id and not os.path.exists(request.speaker_id):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "SPEAKER_NOT_FOUND",
                "message": f"Speaker file not found: {request.speaker_id}",
            },
        )

    # Create task
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    task = await task_manager.create_task(request)

    return CreateTtsTaskResponse(
        request_id=request_id,
        task_id=task["task_id"],
        status=task["status"],
    )


@router.get("/describe_tts_task", response_model=DescribeTtsTaskResponse)
async def describe_tts_task(task_id: str = Query(..., description="Task ID")):
    """Query task status and results."""
    if task_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    request_id = f"req_{uuid.uuid4().hex[:12]}"

    task = await task_manager.get_task(task_id)
    if task is None:
        return DescribeTtsTaskResponse(
            request_id=request_id,
            task=None,
            error={
                "code": "TASK_NOT_FOUND",
                "message": f"Task not found: {task_id}",
            },
        )

    # Build audio/subtitle URLs
    audio_url = None
    subtitle_url = None
    if task["status"] == "success":
        if task.get("audio_path"):
            audio_url = f"/api/v1/files/{task_id}/output.wav"
        if task.get("subtitle_path"):
            subtitle_url = f"/api/v1/files/{task_id}/output.srt"

    task_info = TaskInfo(
        task_id=task["task_id"],
        status=task["status"],
        audio_url=audio_url,
        subtitle_url=subtitle_url,
        duration=task.get("duration"),
        character_count=task.get("character_count"),
        error_code=task.get("error_code"),
        error_message=task.get("error_message"),
        create_time=task["create_time"],
        start_time=task.get("start_time"),
        finish_time=task.get("finish_time"),
    )

    return DescribeTtsTaskResponse(
        request_id=request_id,
        task=task_info,
    )
