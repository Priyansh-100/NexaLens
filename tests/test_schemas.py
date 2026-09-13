import pytest
from uuid import uuid4

from nexalens.core.config import Settings
from nexalens.models.schemas import (
    DataSource,
    DataSourceType,
    Document,
    DocumentChunk,
    QueryIntent,
    QueryRequest,
    QueryResponse,
    SQLResult,
    User,
    UserRole,
)


def test_settings_load():
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://test:test@localhost/test",
        SECRET_KEY="test-secret",
    )
    assert settings.app_name == "nexalens"
    assert settings.llm_model == "llama3.1:8b"


def test_data_source_schema():
    ds = DataSource(
        name="test_db",
        type=DataSourceType.SQL,
        config={"host": "localhost"},
    )
    assert ds.type == DataSourceType.SQL
    assert ds.config["host"] == "localhost"


def test_document_schema():
    doc = Document(
        source_id=uuid4(),
        title="Test Doc",
        content="Content here",
    )
    assert doc.title == "Test Doc"
    assert doc.chunk_count == 0


def test_document_chunk_schema():
    chunk = DocumentChunk(
        document_id=uuid4(),
        content="Chunk content",
        chunk_index=0,
    )
    assert chunk.chunk_index == 0


def test_query_request_schema():
    req = QueryRequest(
        question="What is revenue?",
        max_results=5,
    )
    assert req.question == "What is revenue?"
    assert req.max_results == 5


def test_query_response_schema():
    resp = QueryResponse(
        question="Test",
        intent=QueryIntent.SQL,
        answer="Answer here",
        confidence=0.9,
        processing_time_ms=100.0,
    )
    assert resp.intent == QueryIntent.SQL
    assert resp.confidence == 0.9


def test_sql_result_schema():
    sql_res = SQLResult(
        sql="SELECT 1",
        columns=["col1"],
        rows=[{"col1": 1}],
        row_count=1,
        execution_time_ms=50.0,
    )
    assert sql_res.row_count == 1


def test_user_schema():
    user = User(
        email="test@test.com",
        name="Test User",
        role=UserRole.ANALYST,
        hashed_password="hashed",
    )
    assert user.role == UserRole.ANALYST
    assert user.is_active is True


def test_query_intent_enum():
    assert QueryIntent.SQL.value == "sql"
    assert QueryIntent.DOCUMENT.value == "document"
    assert QueryIntent.HYBRID.value == "hybrid"
    assert QueryIntent.CLARIFICATION.value == "clarification"


def test_user_role_enum():
    assert UserRole.ADMIN.value == "admin"
    assert UserRole.ANALYST.value == "analyst"
    assert UserRole.VIEWER.value == "viewer"