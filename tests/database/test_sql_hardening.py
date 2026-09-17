import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.services.sql_agent import SQLAgent
from nexalens.core.exceptions import SQLGenerationError


class TestSQLHardening:
    @pytest.fixture
    def agent(self):
        return SQLAgent()

    @pytest.fixture
    def allowed_tables(self):
        return {"orders", "customers", "products"}

    def test_validate_sql_rejects_insert(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Forbidden SQL statement"):
            agent.validate_sql("INSERT INTO orders VALUES (1, 2, 3)", allowed_tables)

    def test_validate_sql_rejects_update(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Forbidden SQL statement"):
            agent.validate_sql("UPDATE orders SET amount = 100", allowed_tables)

    def test_validate_sql_rejects_delete(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Forbidden SQL statement"):
            agent.validate_sql("DELETE FROM orders", allowed_tables)

    def test_validate_sql_rejects_drop(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Forbidden SQL statement"):
            agent.validate_sql("DROP TABLE orders", allowed_tables)

    def test_validate_sql_rejects_create(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Forbidden SQL statement"):
            agent.validate_sql("CREATE TABLE test (id INT)", allowed_tables)

    def test_validate_sql_rejects_multi_statement(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Multi-statement queries are not allowed"):
            agent.validate_sql("SELECT * FROM orders; DROP TABLE users;", allowed_tables)

    def test_validate_sql_rejects_disallowed_table(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="not in allowed schema"):
            agent.validate_sql("SELECT * FROM users", allowed_tables)

    def test_validate_sql_rejects_dangerous_function_pg_sleep(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Dangerous function not allowed"):
            agent.validate_sql("SELECT pg_sleep(10)", allowed_tables)

    def test_validate_sql_rejects_dangerous_function_lo_import(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Dangerous function not allowed"):
            agent.validate_sql("SELECT lo_import('/etc/passwd')", allowed_tables)

    def test_validate_sql_rejects_dangerous_function_dblink(self, agent, allowed_tables):
        with pytest.raises(SQLGenerationError, match="Dangerous function not allowed"):
            agent.validate_sql("SELECT * FROM dblink('conn', 'SELECT 1')", allowed_tables)

    def test_validate_sql_rejects_query_too_long(self, agent, allowed_tables):
        long_sql = "SELECT " + ", ".join([f"col{i}" for i in range(10000)]) + " FROM orders"
        with pytest.raises(SQLGenerationError, match="exceeds maximum length"):
            agent.validate_sql(long_sql, allowed_tables)

    def test_validate_sql_allows_valid_select(self, agent, allowed_tables):
        # Should not raise
        agent.validate_sql("SELECT id, amount FROM orders WHERE id = 1", allowed_tables)

    def test_validate_sql_allows_cte(self, agent, allowed_tables):
        # Should not raise
        agent.validate_sql("WITH cte AS (SELECT * FROM orders) SELECT * FROM cte", allowed_tables)

    def test_validate_sql_allows_join(self, agent, allowed_tables):
        # Should not raise
        agent.validate_sql("SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id", allowed_tables)