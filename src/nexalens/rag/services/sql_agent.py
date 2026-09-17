import re
import time
from typing import Any
from uuid import UUID

import sqlglot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.config import get_settings
from nexalens.core.exceptions import SQLExecutionError, SQLGenerationError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import SQLResult
from nexalens.rag.schemas import (
    LLMUsage,
    SQLAgentConfig,
    SQLExecutionMetrics,
    SQLGeneration,
)
from nexalens.services.llm import llm_service
from nexalens.sql.schema import schema_service

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
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "COPY", "CALL", "DO", "COMMIT", "ROLLBACK",
    "SAVEPOINT", "LOCK", "VACUUM", "ANALYZE", "REINDEX", "CLUSTER", "CHECKPOINT",
}

DANGEROUS_FUNCTIONS = {
    "pg_sleep", "lo_import", "lo_export", "dblink", "copy",
    "pg_read_file", "pg_write_file", "pg_ls_dir", "pg_stat_file",
}

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SQLAgent:
    def __init__(self, config: SQLAgentConfig | None = None):
        self.config = config or SQLAgentConfig()
        self.max_rows = self.config.max_rows
        self.timeout = self.config.timeout_seconds
        self.dialect = "postgresql"
        self.max_query_cost = self.config.max_query_cost
        self.max_query_length = self.config.max_query_length
        self.llm = llm_service
        self.schema_service = schema_service

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
    ) -> SQLGeneration:
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
            result = await self._generate_structured(
                prompt=prompt,
                schema=SQLGeneration,
                operation="sql_generation",
            )
            result.sql = self._clean_sql(result.sql)
            self.validate_sql(result.sql, allowed_tables=set(schema.keys()))
            logger.debug("sql_generated", sql=result.sql[:200], tables=result.tables_used)
            return result
        except SQLGenerationError:
            raise
        except Exception as e:
            logger.error("sql_generation_failed", error=str(e), question=question[:100])
            raise SQLGenerationError(f"Failed to generate SQL: {e}") from e

    def _clean_sql(self, sql: str) -> str:
        sql = sql.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip()

    def validate_sql(self, sql: str, allowed_tables: set[str] | None = None) -> None:
        if not sql or not sql.strip():
            raise SQLGenerationError("Empty SQL query")

        if len(sql) > self.max_query_length:
            raise SQLGenerationError(f"Query exceeds maximum length of {self.max_query_length} characters")

        try:
            parsed_list = sqlglot.parse(sql, dialect="postgres")
        except Exception as e:
            raise SQLGenerationError(f"SQL parsing failed: {e}") from e

        if len(parsed_list) > 1:
            raise SQLGenerationError("Multi-statement queries are not allowed")

        parsed = parsed_list[0]

        for statement in parsed.walk():
            stmt_type = type(statement).__name__.upper()
            if stmt_type in FORBIDDEN_KEYWORDS:
                raise SQLGenerationError(f"Forbidden SQL statement: {stmt_type}")

        if not isinstance(parsed, (sqlglot.exp.Select, sqlglot.exp.With)):
            raise SQLGenerationError("Only SELECT/WITH statements are allowed")

        if allowed_tables:
            for table in parsed.find_all(sqlglot.exp.Table):
                table_name = table.name
                if table_name and table_name not in allowed_tables:
                    raise SQLGenerationError(f"Table '{table_name}' not in allowed schema")

        for func in parsed.find_all(sqlglot.exp.Anonymous):
            func_name = func.this.lower()
            if func_name in DANGEROUS_FUNCTIONS:
                raise SQLGenerationError(f"Dangerous function not allowed: {func_name}")

        for func in parsed.find_all(sqlglot.exp.Func):
            func_name = func.__class__.__name__.lower()
            if func_name in DANGEROUS_FUNCTIONS:
                raise SQLGenerationError(f"Dangerous function not allowed: {func_name}")

    async def estimate_query_cost(self, session: AsyncSession, sql: str) -> SQLExecutionMetrics:
        try:
            result = await session.execute(text(f"EXPLAIN (FORMAT JSON) {sql}"))
            plan_data = result.scalar()

            if not plan_data or not isinstance(plan_data, list):
                return SQLExecutionMetrics(
                    sql=sql,
                    execution_time_ms=0,
                    rows_returned=0,
                    columns=[],
                    cost_estimate=0.0,
                    status="error",
                    error="Could not parse plan",
                )

            plan = plan_data[0].get("Plan", {})
            total_cost = plan.get("Total Cost", 0.0)
            estimated_rows = plan.get("Plan Rows", 0)

            def summarize_plan(node: dict, indent: int = 0) -> list[str]:
                lines = []
                prefix = "  " * indent
                node_type = node.get("Node Type", "Unknown")
                cost = node.get("Total Cost", 0)
                rows = node.get("Plan Rows", 0)
                lines.append(f"{prefix}{node_type} (cost={cost:.2f}, rows={rows})")
                for child in node.get("Plans", []):
                    lines.extend(summarize_plan(child, indent + 1))
                return lines

            plan_lines = summarize_plan(plan)
            plan_summary = "\n".join(plan_lines[:10])

            exceeds_limit = total_cost > self.max_query_cost

            return SQLExecutionMetrics(
                sql=sql,
                execution_time_ms=0,
                rows_returned=estimated_rows,
                columns=[],
                cost_estimate=total_cost,
                status="success" if not exceeds_limit else "blocked",
                error="Query cost exceeds limit" if exceeds_limit else None,
            )
        except Exception as e:
            logger.warning("query_cost_estimation_failed", error=str(e))
            return SQLExecutionMetrics(
                sql=sql,
                execution_time_ms=0,
                rows_returned=0,
                columns=[],
                cost_estimate=0.0,
                status="error",
                error=f"Estimation failed: {e}",
            )

    async def execute_sql(
        self,
        session: AsyncSession,
        sql: str,
        params: dict[str, Any] | None = None,
        allowed_tables: set[str] | None = None,
    ) -> SQLResult:
        self.validate_sql(sql, allowed_tables)

        cost_estimate = await self.estimate_query_cost(session, sql)
        if cost_estimate.status == "blocked":
            raise SQLGenerationError(
                f"Query cost {cost_estimate.cost_estimate:.2f} exceeds limit "
                f"{self.max_query_cost:.2f}. Plan: {cost_estimate.error}"
            )

        start_time = time.perf_counter()
        try:
            timeout_ms = self.timeout * 1000
            await session.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))

            result = await session.execute(text(sql), params or {})
            rows = result.mappings().all()
            columns = list(result.keys()) if result.keys() else []
            data = [dict(row) for row in rows]
            execution_time = (time.perf_counter() - start_time) * 1000

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
            if "statement_timeout" in str(e).lower() or "timeout" in str(e).lower():
                raise SQLExecutionError(f"Query timed out after {self.timeout}s") from e
            raise SQLExecutionError(f"SQL execution failed: {e}") from e

    async def explain_sql_result(self, question: str, sql: str, result: SQLResult) -> str:
        prompt = f"""Explain this SQL result in business terms for: "{question}"

SQL: {sql}
Results: {result.row_count} rows, columns: {result.columns}
Sample: {result.rows[:3]}

Provide a 2-3 sentence business interpretation."""
        try:
            return await self.llm.generate(prompt, temperature=0.1, max_tokens=300)
        except Exception:
            return "Query executed successfully."

    async def _generate_structured(
        self,
        prompt: str,
        schema: type[SQLGeneration],
        operation: str,
    ) -> SQLGeneration:
        import json
        format_prompt = f"""Return ONLY valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

No markdown, no explanation, no extra text."""

        full_prompt = f"{prompt}\n\n{format_prompt}"

        response = await self.llm.generate(
            full_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            format="json",
        )

        try:
            data = json.loads(response)
            return schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("structured_sql_output_parse_failed", response=response, error=str(e))
            raise SQLGenerationError(f"Failed to parse structured SQL output: {e}") from e