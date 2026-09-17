import json
from typing import Any
from uuid import UUID

from nexalens.core.config import get_settings
from nexalens.core.exceptions import LLMError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import FinancialModelRequest, FinancialModelResult, FinancialModelType
from nexalens.analytics.financial import financial_modeling_service
from nexalens.rag.schemas import FinancialModelExtraction, LLMUsage
from nexalens.services.llm import llm_service

logger = get_logger(__name__)
settings = get_settings()


class FinancialAnalyzer:
    def __init__(self):
        self.llm = llm_service
        self.financial_service = financial_modeling_service

    async def extract_and_run(
        self,
        question: str,
    ) -> FinancialModelResult | None:
        try:
            extraction = await self._extract_model_params(question)

            fm_request = FinancialModelRequest(
                model_type=extraction.model_type,
                assumptions=extraction.assumptions,
                cash_flows=extraction.cash_flows,
                output_format="excel",
            )

            result = self.financial_service.run_model(fm_request)
            logger.info("financial_model_executed", model_type=result.model_type.value)
            return result

        except Exception as e:
            logger.error("financial_model_failed", error=str(e), question=question[:100])
            return None

    async def _extract_model_params(self, question: str) -> FinancialModelExtraction:
        prompt = f"""Extract financial model parameters from this question:
{question}

Return JSON with:
- model_type: one of [dcf, npv, irr, payback, sensitivity]
- assumptions: dict with keys like discount_rate, terminal_growth, revenue_growth, tax_rate, etc.
- cash_flows: array of numbers or null if not mentioned
- confidence: 0.0 to 1.0

Only include values explicitly mentioned or reasonable defaults."""

        result = await self._generate_structured(
            prompt=prompt,
            schema=FinancialModelExtraction,
            operation="financial_model_extraction",
        )

        logger.debug(
            "financial_model_extracted",
            model_type=result.model_type.value,
            assumptions=result.assumptions,
            cash_flows=result.cash_flows,
        )
        return result

    async def _generate_structured(
        self,
        prompt: str,
        schema: type[FinancialModelExtraction],
        operation: str,
    ) -> FinancialModelExtraction:
        import json
        format_prompt = f"""Return ONLY valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

No markdown, no explanation, no extra text."""

        full_prompt = f"{prompt}\n\n{format_prompt}"

        response = await self.llm.generate(
            full_prompt,
            temperature=0.1,
            max_tokens=300,
            format="json",
        )

        try:
            data = json.loads(response)
            return schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("structured_financial_output_parse_failed", response=response, error=str(e))
            raise LLMError(f"Failed to parse structured financial output: {e}") from e

    def run_model_direct(self, request: FinancialModelRequest) -> FinancialModelResult:
        return self.financial_service.run_model(request)

    def get_assumptions_template(self, model_type: FinancialModelType) -> dict[str, Any]:
        return self.financial_service.get_assumptions_template(model_type)