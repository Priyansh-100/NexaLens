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
        self._readonly_user = settings.postgres_readonly_user if hasattr(settings, 'postgres_readonly_user') else None
        self._readonly_password = settings.postgres_readonly_password if hasattr(settings, 'postgres_readonly_password') else None

    def _build_connection_url(self, config: dict[str, Any], readonly: bool = False) -> URL:
        """Build SQLAlchemy URL from data source config."""
        dialect = config.get("dialect", "postgresql")
        driver = config.get("driver", "asyncpg")

        if readonly and self._readonly_user and self._readonly_password:
            username = self._readonly_user
            password = self._readonly_password
        else:
            username = config.get("user") or config.get("username")
            password = config.get("password")

        return URL.create(
            drivername=f"{dialect}+{driver}",
            username=username,
            password=password,
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database") or config.get("dbname"),
            query=config.get("query", {}),
        )

    def get_engine(self, data_source: DataSourceModel, readonly: bool = True) -> AsyncEngine:
        """Get or create engine for a data source.
        
        Args:
            data_source: The data source model
            readonly: If True, use read-only credentials (default: True for security)
        """
        cache_key = (data_source.id, readonly)
        if cache_key not in self._engines:
            url = self._build_connection_url(data_source.config, readonly=readonly)

            pool_size = settings.database_pool_size if hasattr(settings, 'database_pool_size') else 5
            max_overflow = settings.database_max_overflow if hasattr(settings, 'database_max_overflow') else 10

            engine = create_async_engine(
                url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                poolclass=NullPool if settings.app_env == "testing" else None,
                echo=settings.app_env == "development",
                connect_args={
                    "server_settings": {
                        "default_transaction_read_only": "on" if readonly else "off",
                    }
                } if readonly else {},
            )
            self._engines[cache_key] = engine
            self._sessionmakers[cache_key] = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            logger.info("datasource_engine_created", datasource_id=str(data_source.id), readonly=readonly)

        return self._engines[cache_key]

    def get_sessionmaker(self, data_source: DataSourceModel, readonly: bool = True) -> async_sessionmaker:
        """Get sessionmaker for a data source."""
        self.get_engine(data_source, readonly=readonly)
        cache_key = (data_source.id, readonly)
        return self._sessionmakers[cache_key]

    @asynccontextmanager
    async def get_session(self, data_source: DataSourceModel, readonly: bool = True) -> AsyncSession:
        """Get a session for a data source.
        
        Args:
            data_source: The data source model
            readonly: If True, enforce read-only transaction (default: True)
        """
        sessionmaker = self.get_sessionmaker(data_source, readonly=readonly)
        async with sessionmaker() as session:
            try:
                if readonly:
                    await session.execute(text("SET TRANSACTION READ ONLY"))
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def test_connection(self, data_source: DataSourceModel, readonly: bool = True) -> bool:
        """Test if we can connect to the data source."""
        try:
            engine = self.get_engine(data_source, readonly=readonly)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("datasource_connection_test_passed", datasource_id=str(data_source.id), readonly=readonly)
            return True
        except Exception as e:
            logger.error("datasource_connection_test_failed", datasource_id=str(data_source.id), error=str(e), readonly=readonly)
            return False

    async def close_engine(self, data_source_id: UUID, readonly: bool = True) -> None:
        """Close and remove engine for a data source."""
        cache_key = (data_source_id, readonly)
        if cache_key in self._engines:
            await self._engines[cache_key].dispose()
            del self._engines[cache_key]
            del self._sessionmakers[cache_key]
            logger.info("datasource_engine_closed", datasource_id=str(data_source_id), readonly=readonly)

    async def close_all(self) -> None:
        """Close all engines."""
        for cache_key in list(self._engines.keys()):
            await self.close_engine(cache_key[0], readonly=cache_key[1])


datasource_connection_manager = DataSourceConnectionManager()