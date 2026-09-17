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


class TestRAGOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return RAGOrchestrator()

    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def sample_sql_result(self):
        return SQLResult(
            sql="SELECT SUM(amount) FROM orders",
            columns=["total_revenue"],
            rows=[{"total_revenue": 1000000}],
            row_count=1,
            execution_time_ms=50,
        )

    @pytest.mark.asyncio
    async def test_process_sql_query(self, orchestrator, mock_session, sample_sql_result):
        request = QueryRequest(
            question="What was total revenue in Q3 2023?",
            intent=QueryIntent.SQL,
            data_source_ids=[uuid4()],
        )

        with patch.object(orchestrator.sql_agent, "generate_sql", new=AsyncMock(return_value=MagicMock(
            sql="SELECT SUM(amount) FROM orders WHERE order_date >= '2023-07-01' AND order_date < '2023-10-01'",
            explanation="Q3 revenue",
            tables_used=["orders"],
        ))), \
        patch.object(orchestrator.sql_agent, "execute_sql", new=AsyncMock(return_value=sample_sql_result)), \
        patch.object(orchestrator.sql_agent, "explain_sql_result", new=AsyncMock(return_value="Q3 revenue was $1M")), \
        patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.SQL, confidence=0.9, reasoning="SQL query"
        ))):
            response = await orchestrator.process_query(request, mock_session)
            
            assert response.intent == QueryIntent.SQL
            assert response.sql_result is not None
            assert response.confidence > 0.5

    @pytest.mark.asyncio
    async def test_process_document_query(self, orchestrator, mock_session):
        request = QueryRequest(
            question="What is our refund policy?",
            intent=QueryIntent.DOCUMENT,
        )

        mock_docs = [
            DocumentResult(
                document_id=uuid4(),
                title="Refund Policy",
                chunk_content="Full refund within 30 days",
                score=0.9,
                metadata={},
            )
        ]

        with patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.DOCUMENT, confidence=0.9, reasoning="Document query"
        ))), \
        patch.object(orchestrator.rag_retriever, "search_documents", new=AsyncMock(return_value=mock_docs)):
            response = await orchestrator.process_query(request, mock_session)
            
            assert response.intent == QueryIntent.DOCUMENT
            assert len(response.document_results) == 1
            assert "refund" in response.answer.lower()

    @pytest.mark.asyncio
    async def test_process_financial_model_query(self, orchestrator, mock_session):
        request = QueryRequest(
            question="Calculate NPV of [-100, 30, 40, 50] at 10%",
            intent=QueryIntent.FINANCIAL_MODEL,
        )

        financial_result = FinancialModelResult(
            model_type=FinancialModelType.NPV,
            npv=10.5,
            assumptions_used={"discount_rate": 0.1},
            explanation="NPV positive",
        )

        with patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.FINANCIAL_MODEL, confidence=0.9, reasoning="Financial model"
        ))), \
        patch.object(orchestrator.financial_analyzer, "extract_and_run", new=AsyncMock(return_value=financial_result)):
            response = await orchestrator.process_query(request, mock_session)
            
            assert response.intent == QueryIntent.FINANCIAL_MODEL
            assert response.financial_result is not None
            assert response.financial_result.npv == 10.5

    @pytest.mark.asyncio
    async def test_process_forecast_query(self, orchestrator, mock_session):
        request = QueryRequest(
            question="Forecast revenue for next 12 months",
            intent=QueryIntent.FORECAST,
            data_source_ids=[uuid4()],
        )

        forecast_result = ForecastResult(
            forecast=[ForecastPoint(date="2024-01", yhat=1100000, yhat_lower=1000000, yhat_upper=1200000)],
            metrics={"mae": 50000},
            model_type="prophet",
            parameters={},
            explanation="Revenue forecast",
        )

        with patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.FORECAST, confidence=0.9, reasoning="Forecast"
        ))), \
        patch.object(orchestrator.forecasting_service, "extract_and_forecast", new=AsyncMock(return_value=forecast_result)):
            response = await orchestrator.process_query(request, mock_session)
            
            assert response.intent == QueryIntent.FORECAST
            assert response.forecast_result is not None

    @pytest.mark.asyncio
    async def test_process_schedule_query(self, orchestrator, mock_session):
        request = QueryRequest(
            question="Email me weekly revenue report",
            intent=QueryIntent.SCHEDULE_REPORT,
        )

        schedule_result = ReportSchedule(
            id=uuid4(),
            name="Weekly Revenue",
            query="What was revenue last week?",
            data_source_ids=[uuid4()],
            cron_expression="0 9 * * MON",
            recipients=["test@example.com"],
            format=ReportFormat.PDF,
            is_active=True,
            created_by=uuid4(),
        )

        with patch.object(orchestrator.intent_router, "classify_intent", new=AsyncMock(return_value=MagicMock(
            intent=QueryIntent.SCHEDULE_REPORT, confidence=0.9, reasoning="Schedule"
        ))), \
        patch.object(orchestrator.schedule_extractor, "extract_and_create", new=AsyncMock(return_value=schedule_result)):
            user_id = uuid4()
            response = await orchestrator.process_query(request, mock_session, user_id)
            
            assert response.intent == QueryIntent.SCHEDULE_REPORT
            assert response.schedule_result is not None