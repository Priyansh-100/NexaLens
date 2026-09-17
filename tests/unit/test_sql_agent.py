import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.services.sql_agent import SQLAgent
from nexalens.rag.schemas import SQLGeneration
from nexalens.core.exceptions import SQLGenerationError


class TestSQLAgent:
    @pytest.fixture
    def agent(self):
        return SQLAgent()

    @pytest.fixture
    def sample_schema(self):
        return {
            "orders": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "customer_id", "type": "INTEGER", "nullable": False},
                {"name": "amount", "type": "NUMERIC", "nullable": False},
                {"name": "order_date", "type": "TIMESTAMP", "nullable": False},
            ],
            "customers": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "name", "type": "VARCHAR", "nullable": False},
                {"name": "segment", "type": "VARCHAR", "nullable": True},
            ],
        }

    @pytest.mark.asyncio
    async def test_generate_sql(self, agent, sample_schema):
        with patch.object(agent.llm, "generate_structured", new=AsyncMock(return_value=SQLGeneration(
            sql="SELECT SUM(amount) FROM orders WHERE order_date >= '2023-01-01'",
            explanation="Total revenue for 2023",
            tables_used=["orders"],
        ))):
            result = await agent.generate_sql("What was total revenue in 2023?", sample_schema)
            assert "SELECT" in result.sql
            assert "orders" in result.tables_used

    @pytest.mark.asyncio
    async def test_validate_sql_readonly(self, agent):
        valid_sql = "SELECT * FROM orders WHERE id = 1"
        agent.validate_sql(valid_sql, allowed_tables={"orders"})

    @pytest.mark.asyncio
    async def test_validate_sql_forbidden_keyword(self, agent):
        invalid_sql = "INSERT INTO orders VALUES (1, 2, 3, '2023-01-01')"
        with pytest.raises(SQLGenerationError, match="Forbidden SQL statement"):
            agent.validate_sql(invalid_sql, allowed_tables={"orders"})

    @pytest.mark.asyncio
    async def test_validate_sql_disallowed_table(self, agent):
        sql = "SELECT * FROM users"
        with pytest.raises(SQLGenerationError, match="not in allowed schema"):
            agent.validate_sql(sql, allowed_tables={"orders"})

    @pytest.mark.asyncio
    async def test_validate_sql_multi_statement(self, agent):
        sql = "SELECT * FROM orders; DROP TABLE orders;"
        with pytest.raises(SQLGenerationError, match="Multi-statement queries are not allowed"):
            agent.validate_sql(sql, allowed_tables={"orders"})

    @pytest.mark.asyncio
    async def test_validate_sql_dangerous_function(self, agent):
        sql = "SELECT pg_sleep(10)"
        with pytest.raises(SQLGenerationError, match="Dangerous function not allowed"):
            agent.validate_sql(sql, allowed_tables={"orders"})

    @pytest.mark.asyncio
    async def test_validate_sql_too_long(self, agent):
        long_sql = "SELECT " + "1, " * 10000 + "FROM orders"
        with pytest.raises(SQLGenerationError, match="exceeds maximum length"):
            agent.validate_sql(long_sql, allowed_tables={"orders"})