"""Task management service with MongoDB storage."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..db import Database
from ..models import CreateTtsTaskRequest

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages TTS tasks with MongoDB storage."""

    COLLECTION_NAME = "tasks"

    def __init__(self, output_dir: str):
        """Initialize the task manager."""
        self.output_dir = output_dir

    @property
    def collection(self):
        """Get the tasks collection."""
        return Database.get_db()[self.COLLECTION_NAME]

    async def create_task(self, request: CreateTtsTaskRequest) -> dict:
        """Create a new TTS task."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        task = {
            "task_id": task_id,
            "status": "waiting",
            "text": request.text,
            "language": request.language,
            "speed": request.speed,
            "break_level": request.break_level,
            "speaker_id": request.speaker_id,
            "max_length": request.max_length,
            "max_batch": request.max_batch,
            "skip_subtitles": request.skip_subtitles,
            "whisper_model": request.whisper_model,
            "callback_url": request.callback_url,
            "audio_path": None,
            "subtitle_path": None,
            "duration": None,
            "character_count": len(request.text),
            "error_code": None,
            "error_message": None,
            "create_time": now,
            "start_time": None,
            "finish_time": None,
        }

        await self.collection.insert_one(task)
        logger.info(f"Created task: {task_id}")
        return task

    async def get_task(self, task_id: str) -> Optional[dict]:
        """Get a task by ID."""
        return await self.collection.find_one({"task_id": task_id})

    async def get_next_waiting_task(self) -> Optional[dict]:
        """Get and claim the next waiting task."""
        now = datetime.now(timezone.utc)
        result = await self.collection.find_one_and_update(
            {"status": "waiting"},
            {
                "$set": {
                    "status": "processing",
                    "start_time": now,
                }
            },
            sort=[("create_time", 1)],
            return_document=True,
        )
        if result:
            logger.info(f"Claimed task: {result['task_id']}")
        return result

    async def update_task(self, task_id: str, **updates: Any) -> bool:
        """Update task fields."""
        result = await self.collection.update_one(
            {"task_id": task_id}, {"$set": updates}
        )
        return result.modified_count > 0

    async def complete_task(
        self,
        task_id: str,
        audio_path: str,
        subtitle_path: Optional[str],
        duration: float,
    ) -> bool:
        """Mark a task as completed successfully."""
        now = datetime.now(timezone.utc)
        return await self.update_task(
            task_id,
            status="success",
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            duration=duration,
            finish_time=now,
        )

    async def fail_task(
        self, task_id: str, error_code: str, error_message: str
    ) -> bool:
        """Mark a task as failed."""
        now = datetime.now(timezone.utc)
        logger.error(f"Task {task_id} failed: {error_code} - {error_message}")
        return await self.update_task(
            task_id,
            status="failed",
            error_code=error_code,
            error_message=error_message,
            finish_time=now,
        )

    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove tasks older than max_age_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        result = await self.collection.delete_many(
            {
                "finish_time": {"$lt": cutoff},
                "status": {"$in": ["success", "failed"]},
            }
        )
        if result.deleted_count > 0:
            logger.info(f"Cleaned up {result.deleted_count} old tasks")
        return result.deleted_count

    async def get_task_count(self) -> dict:
        """Get task count by status."""
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        counts = {}
        async for doc in self.collection.aggregate(pipeline):
            counts[doc["_id"]] = doc["count"]
        return counts
