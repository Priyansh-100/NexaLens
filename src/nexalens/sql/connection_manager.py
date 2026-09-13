from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from nexalens.core.config import get_settings
from nexalens.core.exceptions import DatabaseError
from nexalens.core.logging import get_logger
from nexalens.models.database import DataSourceModel

logger = get_logger(__name__)
settings = get_settings()


class DataSourceConnectionManager:
    """Manages connections to external data sources."""

    def __init__(self):
        self._engines: dict[UUID, AsyncEngine] = {}
        self._sessionmakers: dict[UUID, async_sessionmaker] = {}

    def _build_connection_url(self, config: dict[str, Any]) -> URL:
        """Build SQLAlchemy URL from data source config."""
        dialect = config.get("dialect", "postgresql")
        driver = config.get("driver", "asyncpg")

        return URL.create(
            drivername=f"{dialect}+{driver}",
            username=config.get("user") or config.get("username"),
            password=config.get("password"),
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database") or config.get("dbname"),
            query=config.get("query", {}),
        )

    def get_engine(self, data_source: DataSourceModel) -> AsyncEngine:
        """Get or create engine for a data source."""
        if data_source.id not in self._engines:
            url = self._build_connection_url(data_source.config)

            engine = create_async_engine(
                url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                poolclass=NullPool if settings.app_env == "testing" else None,
                echo=settings.app_env == "development",
            )
            self._engines[data_source.id] = engine
            self._sessionmakers[data_source.id] = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            logger.info("datasource_engine_created", datasource_id=str(data_source.id))

        return self._engines[data_source.id]

    def get_sessionmaker(self, data_source: DataSourceModel) -> async_sessionmaker:
        """Get sessionmaker for a data source."""
        self.get_engine(data_source)
        return self._sessionmakers[data_source.id]

    @asynccontextmanager
    async def get_session(self, data_source: DataSourceModel) -> AsyncSession:
        """Get a session for a data source."""
        sessionmaker = self.get_sessionmaker(data_source)
        async with sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def test_connection(self, data_source: DataSourceModel) -> bool:
        """Test if we can connect to the data source."""
        try:
            engine = self.get_engine(data_source)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("datasource_connection_test_passed", datasource_id=str(data_source.id))
            return True
        except Exception as e:
            logger.error("datasource_connection_test_failed", datasource_id=str(data_source.id), error=str(e))
            return False

    async def close_engine(self, data_source_id: UUID) -> None:
        """Close and remove engine for a data source."""
        if data_source_id in self._engines:
            await self._engines[data_source_id].dispose()
            del self._engines[data_source_id]
            del self._sessionmakers[data_source_id]
            logger.info("datasource_engine_closed", datasource_id=str(data_source_id))

    async def close_all(self) -> None:
        """Close all engines."""
        for ds_id in list(self._engines.keys()):
            await self.close_engine(ds_id)


datasource_connection_manager = DataSourceConnectionManager()