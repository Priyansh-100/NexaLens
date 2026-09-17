import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.services.answer_synthesizer import AnswerSynthesizer
from nexalens.rag.schemas import AnswerSynthesis
from nexalens.models.schemas import (
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


class TestAnswerSynthesizer:
    @pytest.fixture
    def synthesizer(self):
        return AnswerSynthesizer()

    @pytest.fixture
    def sample_sql_result(self):
        return SQLResult(
            sql="SELECT SUM(amount) FROM orders",
            columns=["total_revenue"],
            rows=[{"total_revenue": 1000000}],
            row_count=1,
            execution_time_ms=50,
        )

    @pytest.fixture
    def sample_doc_results(self):
        return [
            DocumentResult(
                document_id=uuid4(),
                title="Q3 Report",
                chunk_content="Revenue was $1M in Q3",
                score=0.9,
                metadata={},
            )
        ]

    @pytest.mark.asyncio
    async def test_synthesize_with_financial_result(self, synthesizer):
        financial_result = FinancialModelResult(
            model_type=FinancialModelType.NPV,
            npv=50000,
            assumptions_used={"discount_rate": 0.1},
            explanation="NPV is positive",
        )
        
        result = await synthesizer.synthesize(
            "Calculate NPV",
            sql_result=None,
            doc_results=[],
            financial_result=financial_result,
        )
        
        assert "NPV" in result.answer
        assert "$50,000" in result.answer
        assert result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_synthesize_with_forecast_result(self, synthesizer):
        forecast_result = ForecastResult(
            forecast=[
                ForecastPoint(date="2024-01", yhat=1100000, yhat_lower=1000000, yhat_upper=1200000)
            ],
            metrics={"mae": 50000},
            model_type="prophet",
            parameters={},
            explanation="Revenue forecast",
        )
        
        result = await synthesizer.synthesize(
            "Forecast revenue",
            sql_result=None,
            doc_results=[],
            forecast_result=forecast_result,
        )
        
        assert "1,100,000" in result.answer
        assert "Forecast" in result.answer

    @pytest.mark.asyncio
    async def test_synthesize_with_sql_and_docs(self, synthesizer, sample_sql_result, sample_doc_results):
        result = await synthesizer.synthesize(
            "What was revenue?",
            sql_result=sample_sql_result,
            doc_results=sample_doc_results,
        )
        
        assert "1,000,000" in result.answer or "1000000" in result.answer
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_synthesize_low_confidence_returns_unknown(self, synthesizer):
        result = await synthesizer.synthesize(
            "Unknown question",
            sql_result=None,
            doc_results=[],
        )
        
        assert result.confidence < 0.4
        assert "don't have enough information" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_citations_included(self, synthesizer, sample_sql_result, sample_doc_results):
        result = await synthesizer.synthesize(
            "What was revenue?",
            sql_result=sample_sql_result,
            doc_results=sample_doc_results,
        )
        
        assert len(result.citations) > 0
        assert any(c["type"] == "sql" for c in result.citations)
        assert any(c["type"] == "document" for c in result.citations)