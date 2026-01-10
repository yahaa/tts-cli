"""FastAPI application factory."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from .config import ServerConfig
from .db import Database
from .routes import files as files_module
from .routes import files_router, health_router, tasks_router
from .routes import tasks as tasks_module
from .services.task_manager import TaskManager
from .services.tts_engine import TTSEngine
from .services.worker import CleanupScheduler, TTSWorker

logger = logging.getLogger(__name__)


def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    # Shared state
    state = {
        "task_manager": None,
        "tts_engine": None,
        "worker": None,
        "cleanup_scheduler": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        # Startup
        logger.info("Starting TTS server...")

        # Connect to MongoDB
        await Database.connect(config.mongodb_uri, config.db_name)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "tasks"), exist_ok=True)

        # Initialize services
        state["task_manager"] = TaskManager(config.output_dir)
        state["tts_engine"] = TTSEngine()

        # Set task manager in routes
        tasks_module.set_task_manager(state["task_manager"])
        files_module.set_task_manager(state["task_manager"])
        files_module.set_output_dir(config.output_dir)

        # Preload ChatTTS model
        logger.info("Preloading ChatTTS model...")
        state["tts_engine"].ensure_loaded()

        # Start background worker
        state["worker"] = TTSWorker(
            task_manager=state["task_manager"],
            tts_engine=state["tts_engine"],
            output_dir=config.output_dir,
            num_workers=config.num_workers,
        )
        await state["worker"].start()

        # Start cleanup scheduler
        state["cleanup_scheduler"] = CleanupScheduler(
            task_manager=state["task_manager"],
            cleanup_hours=config.cleanup_hours,
        )
        await state["cleanup_scheduler"].start()

        logger.info(f"TTS server started on {config.host}:{config.port}")

        yield

        # Shutdown
        logger.info("Shutting down TTS server...")

        if state["cleanup_scheduler"]:
            await state["cleanup_scheduler"].stop()
        if state["worker"]:
            await state["worker"].stop()
        await Database.disconnect()

        logger.info("TTS server stopped")

    app = FastAPI(
        title="TTS CLI Server",
        description="Text-to-Speech API server with async task processing",
        version=__version__,
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(files_router, prefix="/api/v1")
    app.include_router(health_router, prefix="/api/v1")

    return app
