from typing import Any
from uuid import UUID

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.logging import get_logger

logger = get_logger(__name__)


class SchemaService:
    def __init__(self):
        self._cache: dict[UUID, dict[str, list[dict[str, Any]]]] = {}

    async def get_schema(self, session: AsyncSession, data_source_id: UUID, use_cache: bool = True) -> dict[str, list[dict[str, Any]]]:
        if use_cache and data_source_id in self._cache:
            return self._cache[data_source_id]

        try:
            inspector = inspect(session.bind)
            tables = inspector.get_table_names()

            schema = {}
            for table in tables:
                columns = inspector.get_columns(table)
                pk_columns = set(inspector.get_pk_constraint(table).get("constrained_columns", []))
                fks = inspector.get_foreign_keys(table)

                schema[table] = []
                for col in columns:
                    col_info = {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"],
                        "default": str(col["default"]) if col["default"] else None,
                        "is_primary_key": col["name"] in pk_columns,
                        "is_foreign_key": any(col["name"] in fk["constrained_columns"] for fk in fks),
                    }
                    schema[table].append(col_info)

            self._cache[data_source_id] = schema
            return schema
        except Exception as e:
            logger.error("schema_introspection_failed", error=str(e), data_source_id=str(data_source_id))
            raise

    async def get_sample_data(self, session: AsyncSession, table: str, limit: int = 3) -> list[dict[str, Any]]:
        try:
            result = await session.execute(text(f"SELECT * FROM {table} LIMIT {limit}"))
            return [dict(row) for row in result.mappings().all()]
        except Exception as e:
            logger.warning("sample_data_failed", error=str(e), table=table)
            return []

    def invalidate_cache(self, data_source_id: UUID) -> None:
        self._cache.pop(data_source_id, None)


schema_service = SchemaService()