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

        # Preload Qwen3-TTS model
        logger.info("Preloading Qwen3-TTS model...")
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

    # OpenAPI tags metadata
    tags_metadata = [
        {
            "name": "tasks",
            "description": "TTS 任务管理接口，用于创建和查询语音合成任务",
        },
        {
            "name": "files",
            "description": "文件下载接口，用于下载生成的音频和字幕文件",
        },
        {
            "name": "health",
            "description": "健康检查接口，用于监控服务状态",
        },
    ]

    app = FastAPI(
        title="TTS Server API",
        description="""
## TTS Server - 文字转语音服务 API

基于 Qwen3-TTS 的异步文字转语音服务，支持中英文语音合成和字幕生成。

### 功能特性

- **异步任务处理**：提交任务后立即返回，通过轮询或回调获取结果
- **中英文支持**：支持中文和英文语音合成
- **字幕生成**：基于 Whisper 自动生成 SRT 字幕
- **音色复用**：支持保存和复用说话人音色

### 使用流程

1. 调用 `POST /api/v1/create_tts_task` 创建任务
2. 调用 `GET /api/v1/describe_tts_task` 查询任务状态
3. 任务完成后，调用 `GET /api/v1/files/{task_id}/output.wav` 下载音频

### 任务状态

| 状态 | 说明 |
|------|------|
| `waiting` | 任务已创建，等待处理 |
| `processing` | 任务正在处理中 |
| `success` | 任务处理成功 |
| `failed` | 任务处理失败 |
        """,
        version=__version__,
        lifespan=lifespan,
        openapi_tags=tags_metadata,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        license_info={
            "name": "AGPLv3+",
            "url": "https://www.gnu.org/licenses/agpl-3.0.html",
        },
        contact={
            "name": "tts-cli",
            "url": "https://github.com/yahaa/tts-cli",
        },
    )

    # Register routes
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(files_router, prefix="/api/v1")
    app.include_router(health_router, prefix="/api/v1")

    return app
