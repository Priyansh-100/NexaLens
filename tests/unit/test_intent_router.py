import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.services.intent_router import IntentRouter
from nexalens.rag.schemas import IntentClassification, QueryIntent
from nexalens.models.schemas import QueryIntent as SchemaQueryIntent


class TestIntentRouter:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    @pytest.mark.asyncio
    async def test_classify_sql_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "sql", "confidence": 0.95, "reasoning": "Query requires SQL aggregation"}')):
            result = await router.classify_intent("What was total revenue last quarter?")
            assert result.intent == QueryIntent.SQL
            assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_classify_document_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "document", "confidence": 0.9, "reasoning": "Policy question"}')):
            result = await router.classify_intent("What is our refund policy?")
            assert result.intent == QueryIntent.DOCUMENT

    @pytest.mark.asyncio
    async def test_classify_hybrid_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "hybrid", "confidence": 0.85, "reasoning": "Needs both SQL and documents"}')):
            result = await router.classify_intent("Show me Q3 revenue and explain anomalies per the quarterly report")
            assert result.intent == QueryIntent.HYBRID

    @pytest.mark.asyncio
    async def test_classify_financial_model_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "financial_model", "confidence": 0.9, "reasoning": "DCF calculation"}')):
            result = await router.classify_intent("Build a DCF model with 15% discount rate")
            assert result.intent == QueryIntent.FINANCIAL_MODEL

    @pytest.mark.asyncio
    async def test_classify_forecast_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "forecast", "confidence": 0.88, "reasoning": "Time series prediction"}')):
            result = await router.classify_intent("Forecast revenue for next 12 months")
            assert result.intent == QueryIntent.FORECAST

    @pytest.mark.asyncio
    async def test_classify_schedule_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "schedule_report", "confidence": 0.9, "reasoning": "Schedule creation"}')):
            result = await router.classify_intent("Email me weekly revenue report")
            assert result.intent == QueryIntent.SCHEDULE_REPORT

    @pytest.mark.asyncio
    async def test_classify_clarification_intent(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(return_value='{"intent": "clarification", "confidence": 0.7, "reasoning": "Ambiguous question"}')):
            result = await router.classify_intent("How are we doing?")
            assert result.intent == QueryIntent.CLARIFICATION

    @pytest.mark.asyncio
    async def test_classify_with_retry(self, router):
        with patch.object(router.llm, "generate", new=AsyncMock(side_effect=[
            Exception("First attempt failed"),
            '{"intent": "sql", "confidence": 0.9, "reasoning": "SQL query"}'
        ])):
            result = await router.classify_with_retry("What was revenue?")
            assert result.intent == QueryIntent.SQL