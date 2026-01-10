"""File download routes."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.task_manager import TaskManager

router = APIRouter(tags=["files"])

# Will be set by app.py during startup
task_manager: TaskManager = None
output_dir: str = "./tts_output"


def set_task_manager(manager: TaskManager) -> None:
    """Set the task manager instance."""
    global task_manager
    task_manager = manager


def set_output_dir(path: str) -> None:
    """Set the output directory."""
    global output_dir
    output_dir = path


@router.get("/files/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """Download generated audio or subtitle file."""
    if task_manager is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    # Validate filename
    if filename not in ("output.wav", "output.srt"):
        raise HTTPException(status_code=404, detail="File not found")

    # Check task exists and is completed
    task = await task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "TASK_NOT_FOUND", "message": f"Task not found: {task_id}"},
        )

    if task["status"] != "success":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "TASK_NOT_COMPLETED",
                "message": f"Task not completed: status={task['status']}",
            },
        )

    # Build file path
    file_path = os.path.join(output_dir, "tasks", task_id, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    media_type = "audio/wav" if filename.endswith(".wav") else "text/plain"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )
