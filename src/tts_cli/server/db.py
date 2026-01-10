"""MongoDB connection management."""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database connection manager."""

    client: Optional[AsyncIOMotorClient] = None
    _db_name: str = "tts-server"

    @classmethod
    async def connect(cls, uri: str, db_name: str = "tts-server") -> None:
        """Connect to MongoDB."""
        cls._db_name = db_name
        cls.client = AsyncIOMotorClient(uri)
        # Verify connection
        try:
            await cls.client.admin.command("ping")
            logger.info(f"Connected to MongoDB: {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    @classmethod
    async def disconnect(cls) -> None:
        """Disconnect from MongoDB."""
        if cls.client:
            cls.client.close()
            cls.client = None
            logger.info("Disconnected from MongoDB")

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if cls.client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return cls.client[cls._db_name]

    @classmethod
    def is_connected(cls) -> bool:
        """Check if database is connected."""
        return cls.client is not None
