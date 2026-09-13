import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.orchestrator import RAGOrchestrator
from nexalens.models.schemas import (
    FinancialModelResult,
    FinancialModelType,
    ForecastResult,
    ForecastPoint,
    QueryIntent,
    QueryRequest,
    QueryResponse,
    ReportSchedule,
    ReportFormat,
    SQLResult,
    UserRole,
)
from nexalens.models.database import UserModel


class TestRAGOrchestratorNewIntents:
    @pytest.fixture
    def orchestrator(self):
        return RAGOrchestrator()

    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def mock_user(self):
        return UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )

    @pytest.mark.asyncio
    async def test_classify_financial_model_intent(self, orchestrator):
        with patch.object(orchestrator.llm_service, "generate", new=AsyncMock(return_value="financial_model")):
            intent = await orchestrator.classify_intent("Build a DCF model with 15% discount rate")
            assert intent == QueryIntent.FINANCIAL_MODEL

    @pytest.mark.asyncio
    async def test_classify_forecast_intent(self, orchestrator):
        with patch.object(orchestrator.llm_service, "generate", new=AsyncMock(return_value="forecast")):
            intent = await orchestrator.classify_intent("Forecast revenue for next 12 months")
            assert intent == QueryIntent.FORECAST

    @pytest.mark.asyncio
    async def test_classify_schedule_report_intent(self, orchestrator):
        with patch.object(orchestrator.llm_service, "generate", new=AsyncMock(return_value="schedule_report")):
            intent = await orchestrator.classify_intent("Email me weekly revenue report")
            assert intent == QueryIntent.SCHEDULE_REPORT

    @pytest.mark.asyncio
    async def test_execute_financial_model(self, orchestrator, mock_session):
        request = QueryRequest(question="Calculate NPV of [-100, 30, 40, 50] at 10%")

        with patch.object(orchestrator, "_extract_model_type", new=AsyncMock(return_value=FinancialModelType.NPV)), \
             patch.object(orchestrator, "_extract_assumptions", new=AsyncMock(return_value={"discount_rate": 0.1})), \
             patch.object(orchestrator, "_extract_cash_flows", new=AsyncMock(return_value=[-100, 30, 40, 50])):

            result = await orchestrator._execute_financial_model(request, mock_session)

        assert result is not None
        assert isinstance(result, FinancialModelResult)
        assert result.model_type == FinancialModelType.NPV

    @pytest.mark.asyncio
    async def test_execute_forecast(self, orchestrator, mock_session):
        from nexalens.models.schemas import SQLResult
        import pandas as pd

        request = QueryRequest(
            question="Forecast revenue",
            data_source_ids=[uuid4()],
        )

        mock_schema = {"orders": [{"name": "order_date", "type": "TIMESTAMP"}, {"name": "amount", "type": "NUMERIC"}]}

        with patch.object(orchestrator.schema_service, "get_schema", new=AsyncMock(return_value=mock_schema)), \
             patch.object(orchestrator, "_identify_timeseries_table", new=AsyncMock(return_value=("orders", "amount", "order_date"))), \
             patch.object(orchestrator.sql_service, "execute_sql", new=AsyncMock(return_value=SQLResult(
                 sql="SELECT order_date, amount FROM orders ORDER BY order_date",
                 columns=["order_date", "amount"],
                 rows=[{"order_date": f"2023-{i:02d}-01", "amount": 1000 + i * 100} for i in range(12)],
                 row_count=12,
                 execution_time_ms=50,
             ))):

            result = await orchestrator._execute_forecast(request, mock_session)

        assert result is not None
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) > 0

    @pytest.mark.asyncio
    async def test_create_schedule(self, orchestrator, mock_session, mock_user):
        request = QueryRequest(question="Schedule weekly report")

        with patch.object(orchestrator, "_extract_schedule_params", new=AsyncMock(return_value={
            "name": "Weekly Report",
            "query": "What was revenue last week?",
            "data_source_ids": [uuid4()],
            "cron_expression": "0 9 * * MON",
            "recipients": ["test@example.com"],
            "format": "pdf",
        })) as mock_extract:

            result = await orchestrator._create_schedule(request, mock_session, mock_user.id)

        assert result is not None
        assert isinstance(result, ReportSchedule)
        assert result.name == "Weekly Report"

    @pytest.mark.asyncio
    async def test_process_query_financial_model(self, orchestrator, mock_session):
        request = QueryRequest(
            question="Calculate NPV of project",
            intent=QueryIntent.FINANCIAL_MODEL,
        )

        with patch.object(orchestrator, "_execute_financial_model", new=AsyncMock(return_value=FinancialModelResult(
            model_type=FinancialModelType.NPV,
            npv=50000,
            assumptions_used={},
            explanation="NPV is positive",
        ))), \
        patch.object(orchestrator, "_calculate_confidence", return_value=0.9):

            response = await orchestrator.process_query(request, mock_session)

        assert response.intent == QueryIntent.FINANCIAL_MODEL
        assert response.financial_result is not None
        assert response.financial_result.npv == 50000

    @pytest.mark.asyncio
    async def test_process_query_forecast(self, orchestrator, mock_session):
        request = QueryRequest(
            question="Forecast revenue",
            intent=QueryIntent.FORECAST,
        )

        with patch.object(orchestrator, "_execute_forecast", new=AsyncMock(return_value=ForecastResult(
            forecast=[ForecastPoint(date="2024-01", yhat=10000, yhat_lower=9000, yhat_upper=11000)],
            metrics={"mae": 500},
            model_type="prophet",
            parameters={},
            explanation="Forecast generated",
        ))), \
        patch.object(orchestrator, "_calculate_confidence", return_value=0.85):

            response = await orchestrator.process_query(request, mock_session)

        assert response.intent == QueryIntent.FORECAST
        assert response.forecast_result is not None

    @pytest.mark.asyncio
    async def test_process_query_schedule_report(self, orchestrator, mock_session, mock_user):
        request = QueryRequest(
            question="Schedule weekly report",
            intent=QueryIntent.SCHEDULE_REPORT,
        )

        with patch.object(orchestrator, "_create_schedule", new=AsyncMock(return_value=ReportSchedule(
            id=uuid4(),
            name="Weekly Report",
            query="Weekly revenue",
            data_source_ids=[uuid4()],
            cron_expression="0 9 * * MON",
            recipients=["test@example.com"],
            format=ReportFormat.PDF,
            is_active=True,
            created_by=mock_user.id,
        ))), \
        patch.object(orchestrator, "_calculate_confidence", return_value=0.95):

            response = await orchestrator.process_query(request, mock_session, mock_user.id)

        assert response.intent == QueryIntent.SCHEDULE_REPORT
        assert response.schedule_result is not None