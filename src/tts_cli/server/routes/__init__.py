"""API routes."""

from .files import router as files_router
from .health import router as health_router
from .tasks import router as tasks_router

__all__ = ["tasks_router", "files_router", "health_router"]
