"""TTS CLI Server module."""

import logging

import uvicorn

from .app import create_app
from .config import ServerConfig

__all__ = ["run_server", "ServerConfig", "create_app", "cli"]


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    mongodb_uri: str = "mongodb://localhost:27017",
    db_name: str = "tts-server",
    output_dir: str = "./tts_output",
    num_workers: int = 1,
    cleanup_hours: int = 24,
    log_level: str = "info",
) -> None:
    """Run the TTS server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create config
    config = ServerConfig(
        host=host,
        port=port,
        mongodb_uri=mongodb_uri,
        db_name=db_name,
        output_dir=output_dir,
        num_workers=num_workers,
        cleanup_hours=cleanup_hours,
    )

    # Create app
    app = create_app(config)

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
