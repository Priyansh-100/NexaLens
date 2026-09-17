import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.services.llm import LLMService
from nexalens.rag.schemas import IntentClassification, SQLGeneration, QueryIntent


class TestLLMMocked:
    @pytest.fixture
    def llm_service(self):
        return LLMService()

    @pytest.mark.asyncio
    async def test_generate_structured_intent_classification(self, llm_service):
        mock_response = '{"intent": "sql", "confidence": 0.95, "reasoning": "Query requires SQL aggregation"}'
        
        with patch.object(llm_service.client, "chat", new=AsyncMock(return_value={"message": {"content": mock_response}})):
            result = await llm_service.generate_structured(
                prompt="Classify: What was revenue?",
                schema_class=IntentClassification,
                temperature=0.0,
            )
            
            assert result.intent == QueryIntent.SQL
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_generate_structured_sql_generation(self, llm_service):
        mock_response = '{"sql": "SELECT SUM(amount) FROM orders", "explanation": "Total revenue", "tables_used": ["orders"], "estimated_complexity": "low"}'
        
        with patch.object(llm_service.client, "chat", new=AsyncMock(return_value={"message": {"content": mock_response}})):
            result = await llm_service.generate_structured(
                prompt="Generate SQL: What was revenue?",
                schema_class=SQLGeneration,
                temperature=0.0,
            )
            
            assert "SELECT" in result.sql
            assert "orders" in result.tables_used

    @pytest.mark.asyncio
    async def test_generate_structured_retry_on_json_error(self, llm_service):
        # First call returns invalid JSON, second returns valid
        mock_responses = [
            {"message": {"content": "invalid json"}},
            {"message": {"content": '{"intent": "sql", "confidence": 0.9, "reasoning": "Fixed"}'}},
        ]
        
        call_count = 0
        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            resp = mock_responses[call_count]
            call_count += 1
            return resp
        
        with patch.object(llm_service.client, "chat", new=mock_chat):
            result = await llm_service.generate_structured(
                prompt="Classify",
                schema_class=IntentClassification,
                temperature=0.0,
                max_retries=3,
            )
            
            assert result.intent == QueryIntent.SQL
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_structured_retry_on_validation_error(self, llm_service):
        # First call returns valid JSON but fails validation, second succeeds
        mock_responses = [
            {"message": {"content": '{"intent": "invalid_intent", "confidence": 0.9, "reasoning": "Bad"}'}},
            {"message": {"content": '{"intent": "sql", "confidence": 0.9, "reasoning": "Fixed"}'}},
        ]
        
        call_count = 0
        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            resp = mock_responses[call_count]
            call_count += 1
            return resp
        
        with patch.object(llm_service.client, "chat", new=mock_chat):
            result = await llm_service.generate_structured(
                prompt="Classify",
                schema_class=IntentClassification,
                temperature=0.0,
                max_retries=3,
            )
            
            assert result.intent == QueryIntent.SQL
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_temperature_zero_respected(self, llm_service):
        with patch.object(llm_service.client, "chat", new=AsyncMock(return_value={"message": {"content": "test"}})) as mock_chat:
            await llm_service.generate("test prompt", temperature=0.0)
            
            # Verify temperature=0.0 was passed
            call_args = mock_chat.call_args
            options = call_args[1].get("options", {})
            assert options.get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_temperature_none_uses_default(self, llm_service):
        llm_service.temperature = 0.5
        with patch.object(llm_service.client, "chat", new=AsyncMock(return_value={"message": {"content": "test"}})) as mock_chat:
            await llm_service.generate("test prompt", temperature=None)
            
            call_args = mock_chat.call_args
            options = call_args[1].get("options", {})
            assert options.get("temperature") == 0.5