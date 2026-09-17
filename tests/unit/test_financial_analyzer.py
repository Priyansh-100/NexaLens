import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.services.financial_analyzer import FinancialAnalyzer
from nexalens.rag.schemas import FinancialModelExtraction
from nexalens.models.schemas import FinancialModelType


class TestFinancialAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FinancialAnalyzer()

    @pytest.mark.asyncio
    async def test_extract_dcf_model(self, analyzer):
        with patch.object(analyzer.llm, "generate_structured", new=AsyncMock(return_value=FinancialModelExtraction(
            model_type=FinancialModelType.DCF,
            assumptions={"discount_rate": 0.12, "terminal_growth": 0.03},
            cash_flows=[-100000, 30000, 40000, 50000],
            confidence=0.9,
        ))):
            result = await analyzer._extract_model_params("Build a DCF model with 15% discount rate and 3% terminal growth")
            assert result.model_type == FinancialModelType.DCF
            assert result.assumptions["discount_rate"] == 0.12
            assert result.cash_flows == [-100000, 30000, 40000, 50000]

    @pytest.mark.asyncio
    async def test_extract_npv_model(self, analyzer):
        with patch.object(analyzer.llm, "generate_structured", new=AsyncMock(return_value=FinancialModelExtraction(
            model_type=FinancialModelType.NPV,
            assumptions={"discount_rate": 0.1},
            cash_flows=[-100000, 30000, 40000, 50000],
            confidence=0.95,
        ))):
            result = await analyzer._extract_model_params("Calculate NPV of cash flows [-100000, 30000, 40000, 50000] at 10% discount")
            assert result.model_type == FinancialModelType.NPV

    @pytest.mark.asyncio
    async def test_run_model_direct(self, analyzer):
        from nexalens.models.schemas import FinancialModelRequest
        request = FinancialModelRequest(
            model_type=FinancialModelType.NPV,
            assumptions={"discount_rate": 0.1},
            cash_flows=[-100000, 30000, 40000, 50000],
        )
        result = analyzer.run_model_direct(request)
        assert result.npv is not None
        assert result.model_type == FinancialModelType.NPV