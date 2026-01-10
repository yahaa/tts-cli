"""Background worker for processing TTS tasks."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from .task_manager import TaskManager
from .tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class TTSWorker:
    """Background worker for processing TTS tasks."""

    def __init__(
        self,
        task_manager: TaskManager,
        tts_engine: TTSEngine,
        output_dir: str,
        num_workers: int = 1,
    ):
        self.task_manager = task_manager
        self.tts_engine = tts_engine
        self.output_dir = output_dir
        self.num_workers = num_workers
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    async def start(self) -> None:
        """Start background workers."""
        self._running = True
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._tasks.append(task)
        logger.info(f"Started {self.num_workers} TTS worker(s)")

    async def stop(self) -> None:
        """Stop background workers."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._executor.shutdown(wait=True)
        logger.info("TTS workers stopped")

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop."""
        logger.info(f"Worker {worker_id} started")
        while self._running:
            try:
                # Get next waiting task
                task = await self.task_manager.get_next_waiting_task()
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1)
                    continue

                logger.info(f"Worker {worker_id} processing task: {task['task_id']}")

                # Process the task in thread pool (CPU-bound)
                loop = asyncio.get_event_loop()
                try:
                    audio_path, subtitle_path, duration = await loop.run_in_executor(
                        self._executor,
                        self._process_task,
                        task,
                    )

                    # Mark task as completed
                    await self.task_manager.complete_task(
                        task["task_id"],
                        audio_path=audio_path,
                        subtitle_path=subtitle_path,
                        duration=duration,
                    )
                    logger.info(
                        f"Task {task['task_id']} completed: {duration:.2f}s audio"
                    )

                except Exception as e:
                    logger.exception(f"Task {task['task_id']} failed")
                    await self.task_manager.fail_task(
                        task["task_id"],
                        error_code="GENERATION_FAILED",
                        error_message=str(e),
                    )

                # Send callback if URL provided
                if task.get("callback_url"):
                    await self._send_callback(task["task_id"], task["callback_url"])

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(f"Worker {worker_id} error")
                await asyncio.sleep(5)

    def _process_task(self, task: dict) -> tuple[str, Optional[str], float]:
        """Process a single task (runs in thread pool)."""
        return self.tts_engine.generate(
            text=task["text"],
            output_dir=self.output_dir,
            task_id=task["task_id"],
            language=task["language"],
            speed=task["speed"],
            break_level=task["break_level"],
            speaker_id=task.get("speaker_id"),
            max_length=task["max_length"],
            max_batch=task.get("max_batch", 1),
            skip_subtitles=task["skip_subtitles"],
            whisper_model=task["whisper_model"],
        )

    async def _send_callback(self, task_id: str, callback_url: str) -> None:
        """Send task result to callback URL."""
        task = await self.task_manager.get_task(task_id)
        if not task:
            return

        payload = {
            "task_id": task["task_id"],
            "status": task["status"],
            "audio_url": f"/api/v1/files/{task_id}/output.wav"
            if task.get("audio_path")
            else None,
            "subtitle_url": f"/api/v1/files/{task_id}/output.srt"
            if task.get("subtitle_path")
            else None,
            "duration": task.get("duration"),
            "error_code": task.get("error_code"),
            "error_message": task.get("error_message"),
            "finish_time": task["finish_time"].isoformat()
            if task.get("finish_time")
            else None,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    callback_url,
                    json=payload,
                    timeout=30.0,
                )
                logger.info(f"Callback sent for task {task_id}: {response.status_code}")
        except Exception as e:
            logger.warning(f"Callback failed for task {task_id}: {e}")


class CleanupScheduler:
    """Periodic cleanup of old tasks."""

    def __init__(
        self,
        task_manager: TaskManager,
        cleanup_hours: int = 24,
        interval_minutes: int = 60,
    ):
        self.task_manager = task_manager
        self.cleanup_hours = cleanup_hours
        self.interval_minutes = interval_minutes
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start cleanup scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            f"Cleanup scheduler started (every {self.interval_minutes} minutes)"
        )

    async def stop(self) -> None:
        """Stop cleanup scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_minutes * 60)
                removed = await self.task_manager.cleanup_old_tasks(self.cleanup_hours)
                if removed > 0:
                    logger.info(f"Cleaned up {removed} old tasks")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cleanup error")
