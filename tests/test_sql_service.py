import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexalens.sql.service import SQLService


@pytest.fixture
def sql_service():
    return SQLService()


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
def sample_schema():
    return {
        "orders": [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "customer_id", "type": "INTEGER", "nullable": False},
            {"name": "amount", "type": "NUMERIC(10,2)", "nullable": False},
            {"name": "order_date", "type": "TIMESTAMP", "nullable": False},
        ],
        "customers": [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR(255)", "nullable": False},
            {"name": "email", "type": "VARCHAR(255)", "nullable": True},
        ],
    }


class TestSQLService:
    def test_build_schema_context(self, sql_service, sample_schema):
        context = sql_service._build_schema_context(sample_schema)
        assert "TABLE: orders" in context
        assert "TABLE: customers" in context
        assert "id: INTEGER" in context
        assert "amount: NUMERIC(10,2)" in context

    def test_clean_sql(self, sql_service):
        dirty = "```sql\nSELECT * FROM users\n```"
        clean = sql_service._clean_sql(dirty)
        assert clean == "SELECT * FROM users"

        dirty2 = "  SELECT * FROM users  "
        clean2 = sql_service._clean_sql(dirty2)
        assert clean2 == "SELECT * FROM users"

    @pytest.mark.asyncio
    async def test_generate_sql(self, sql_service, sample_schema):
        with patch("nexalens.sql.service.llm_service") as mock_llm:
            mock_llm.generate = AsyncMock(return_value="SELECT SUM(amount) FROM orders")
            sql = await sql_service.generate_sql("What is total revenue?", sample_schema)
            assert "SELECT" in sql.upper()

    @pytest.mark.asyncio
    async def test_execute_sql(self, sql_service, mock_session):
        mock_result = MagicMock()
        mock_result.keys.return_value = ["total"]
        mock_result.mappings.return_value.all.return_value = [{"total": 1000}]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await sql_service.execute_sql(mock_session, "SELECT 1")
        assert result.row_count == 1
        assert result.columns == ["total"]
        assert result.rows == [{"total": 1000}]


class TestSQLCleaning:
    @pytest.mark.parametrize("input_sql,expected", [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 1", "SELECT 1"),
        ("  SELECT 1  ", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
    ])
    def test_clean_variations(self, sql_service, input_sql, expected):
        assert sql_service._clean_sql(input_sql) == expected