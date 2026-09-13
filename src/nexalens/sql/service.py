import re
import time
from typing import Any
from uuid import UUID

import pandas as pd
import sqlglot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.config import get_settings
from nexalens.core.exceptions import SQLExecutionError, SQLGenerationError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import SQLResult
from nexalens.services.llm import llm_service

logger = get_logger(__name__)
settings = get_settings()


SQL_SYSTEM_PROMPT = """You are an expert SQL analyst. Generate precise, executable SQL for PostgreSQL.

Rules:
1. Return ONLY valid SQL - no markdown, no explanations
2. Use explicit column names, never SELECT *
3. Use proper table aliases
4. Handle NULLs correctly with COALESCE
5. Limit results to {max_rows} rows
6. Use CTEs for complex queries
7. Add comments for complex logic
8. Use parameterized queries for values
9. Cast types explicitly when needed
10. Optimize for readability and performance
11. ONLY generate SELECT statements (read-only)
12. Do NOT use: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, COPY, CALL, DO

Schema Context:
{schema}

Example:
Question: "Show total revenue by month for 2023"
SQL:
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total_revenue
    FROM orders
    WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01'
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT month, total_revenue FROM monthly_revenue ORDER BY month;"""


FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "COPY",
    "CALL",
    "DO",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "LOCK",
    "VACUUM",
    "ANALYZE",
    "REINDEX",
    "CLUSTER",
    "CHECKPOINT",
}


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SQLService:
    def __init__(self):
        self.max_rows = settings.sql_max_rows
        self.timeout = settings.sql_timeout
        self.dialect = settings.sql_dialect

    def _build_schema_context(self, tables: dict[str, list[dict[str, Any]]]) -> str:
        lines = []
        for table_name, columns in tables.items():
            lines.append(f"TABLE: {table_name}")
            for col in columns:
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                default = f" DEFAULT {col['default']}" if col.get("default") else ""
                lines.append(f"  - {col['name']}: {col['type']} {nullable}{default}")
            lines.append("")
        return "\n".join(lines)

    async def generate_sql(
        self,
        question: str,
        schema: dict[str, list[dict[str, Any]]],
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> str:
        schema_context = self._build_schema_context(schema)

        prompt_parts = [
            SQL_SYSTEM_PROMPT.format(max_rows=self.max_rows, schema=schema_context),
            f"Question: {question}",
        ]

        if few_shot_examples:
            prompt_parts.insert(1, "Examples:")
            for ex in few_shot_examples:
                prompt_parts.append(f"Q: {ex['question']}\nSQL:\n{ex['sql']}\n")

        prompt = "\n".join(prompt_parts) + "\nSQL:"

        try:
            sql = await llm_service.generate(prompt, temperature=0.0, max_tokens=2048)
            sql = self._clean_sql(sql)
            return sql
        except Exception as e:
            logger.error("sql_generation_failed", error=str(e), question=question)
            raise SQLGenerationError(f"Failed to generate SQL: {e}") from e

    def _clean_sql(self, sql: str) -> str:
        sql = sql.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip()

    def validate_sql(self, sql: str, allowed_tables: set[str] | None = None) -> None:
        """Validate SQL for safety: read-only, no forbidden keywords, valid identifiers."""
        if not sql or not sql.strip():
            raise SQLGenerationError("Empty SQL query")

        # Parse with sqlglot for proper AST analysis
        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
        except Exception as e:
            raise SQLGenerationError(f"SQL parsing failed: {e}") from e

        # Check for forbidden statement types
        for statement in parsed.walk():
            stmt_type = type(statement).__name__.upper()
            if stmt_type in FORBIDDEN_KEYWORDS:
                raise SQLGenerationError(f"Forbidden SQL statement: {stmt_type}")

        # Ensure it's a SELECT or WITH statement
        if not isinstance(parsed, (sqlglot.exp.Select, sqlglot.exp.With)):
            raise SQLGenerationError("Only SELECT/WITH statements are allowed")

        # Validate identifiers if allowed_tables provided
        if allowed_tables:
            for table in parsed.find_all(sqlglot.exp.Table):
                table_name = table.name
                if table_name and table_name not in allowed_tables:
                    raise SQLGenerationError(f"Table '{table_name}' not in allowed schema")

    async def execute_sql(
        self,
        session: AsyncSession,
        sql: str,
        params: dict[str, Any] | None = None,
        allowed_tables: set[str] | None = None,
    ) -> SQLResult:
        # Validate SQL before execution
        self.validate_sql(sql, allowed_tables)

        start_time = time.perf_counter()
        try:
            # Apply statement timeout
            timeout_ms = self.timeout * 1000
            await session.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))

            result = await session.execute(text(sql), params or {})
            rows = result.mappings().all()
            columns = list(result.keys()) if result.keys() else []
            data = [dict(row) for row in rows]
            execution_time = (time.perf_counter() - start_time) * 1000

            # Truncate to max_rows
            truncated_data = data[: self.max_rows]

            return SQLResult(
                sql=sql,
                columns=columns,
                rows=truncated_data,
                row_count=len(data),
                execution_time_ms=execution_time,
            )
        except sqlglot.errors.ParseError as e:
            logger.error("sql_validation_failed", error=str(e), sql=sql[:200])
            raise SQLGenerationError(f"SQL validation failed: {e}") from e
        except SQLGenerationError:
            raise
        except Exception as e:
            logger.error("sql_execution_failed", error=str(e), sql=sql[:200])
            raise SQLExecutionError(f"SQL execution failed: {e}") from e

    async def explain_sql(self, session: AsyncSession, sql: str) -> str:
        try:
            result = await session.execute(text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"))
            return "\n".join(row[0] for row in result.fetchall())
        except Exception as e:
            logger.warning("sql_explain_failed", error=str(e))
            return f"Could not explain query: {e}"

    def to_dataframe(self, result: SQLResult) -> pd.DataFrame:
        return pd.DataFrame(result.rows, columns=result.columns)


sql_service = SQLService()