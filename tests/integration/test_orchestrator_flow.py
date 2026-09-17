import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.orchestrator import RAGOrchestrator
from nexalens.models.schemas import (
    QueryRequest,
    QueryIntent,
    SQLResult,
    DocumentResult,
    FinancialModelResult,
    FinancialModelType,
    ForecastResult,
    ForecastPoint,
    ForecastModelType,
    ReportSchedule,
    ReportFormat,
)


class TestOrchestratorIntegration:
    @pytest.fixture
    def orchestrator(self):
        return RAGOrchestrator()

    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_hybrid_query_sql_and_documents(self, orchestrator, mock_session):
        request = QueryRequest(
            question="Show me Q3 revenue and explain anomalies per the quarterly report",
            intent=QueryIntent.HYBRID,
            data_source_ids=[uuid4()],
        )

        sample_sql = SQLResult(
            sql="SELECT SUM(amount) FROM orders WHERE order_date >= '2023-07-01' AND order_date < '2023-10-01'",
            columns=["total_revenue"],
            rows=[{"total_revenue": 1000000}],
            row_count=1,
            execution_time_ms=50,
        )

        mock_docs = [
            DocumentResult(
                document_id=uuid4(),
                title="Q3 Board Report",
                chunk_content="Revenue missed target by 5% due to supply chain issues",
                score=0.85,
                metadata={},
            )
        ]

        with patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.HYBRID, confidence=0.9, reasoning="Hybrid query"
        ))), \
        patch.object(orchestrator.sql_agent, "generate_sql", new=AsyncMock(return_value=MagicMock(
            sql="SELECT SUM(amount) FROM orders WHERE order_date >= '2023-07-01' AND order_date < '2023-10-01'",
            explanation="Q3 revenue",
            tables_used=["orders"],
        ))), \
        patch.object(orchestrator.sql_agent, "execute_sql", new=AsyncMock(return_value=sample_sql)), \
        patch.object(orchestrator.sql_agent, "explain_sql_result", new=AsyncMock(return_value="Q3 revenue was $1M")), \
        patch.object(orchestrator.rag_retriever, "search_documents", new=AsyncMock(return_value=mock_docs)):
            response = await orchestrator.process_query(request, mock_session)
            
            assert response.intent == QueryIntent.HYBRID
            assert response.sql_result is not None
            assert len(response.document_results) == 1
            assert "supply chain" in response.answer.lower()

    @pytest.mark.asyncio
    async def test_clarification_intent(self, orchestrator, mock_session):
        request = QueryRequest(
            question="How are we doing?",
            intent=QueryIntent.CLARIFICATION,
        )

        with patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.CLARIFICATION, confidence=0.7, reasoning="Ambiguous"
        ))):
            response = await orchestrator.process_query(request, mock_session)
            
            assert response.intent == QueryIntent.CLARIFICATION
            assert response.confidence < 0.5