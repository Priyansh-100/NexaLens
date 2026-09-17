import json
from typing import Any
from uuid import UUID

from nexalens.core.config import get_settings
from nexalens.core.exceptions import LLMError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import QueryIntent
from nexalens.rag.schemas import (
    IntentClassification,
    IntentRouterConfig,
    LLMUsage,
)
from nexalens.services.llm import llm_service

logger = get_logger(__name__)
settings = get_settings()

INTENT_CLASSIFIER_PROMPT = """Classify the user's question into one of these intents:
- sql: Requires querying structured database (metrics, aggregations, filters, joins)
- document: Requires searching unstructured documents (policies, reports, contracts)
- hybrid: Needs both structured data AND document context
- clarification: Ambiguous, needs more info
- financial_model: Financial calculations (DCF, NPV, IRR, payback, sensitivity analysis)
- forecast: Time-series forecasting (revenue, churn, demand, etc.)
- schedule_report: Create or manage scheduled report delivery

Examples:
Q: "What was total revenue last quarter?" -> sql
Q: "What's our refund policy?" -> document
Q: "Show me Q3 revenue and explain any anomalies per the quarterly report" -> hybrid
Q: "How are we doing?" -> clarification
Q: "Build a DCF model with 15% discount rate and 3% terminal growth" -> financial_model
Q: "Calculate NPV of cash flows [-100, 30, 40, 50] at 10% discount" -> financial_model
Q: "Forecast revenue for next 12 months" -> forecast
Q: "Predict churn rate for next quarter using Prophet" -> forecast
Q: "Email me monthly revenue report every Monday 9am" -> schedule_report
Q: "Schedule weekly dashboard delivery to Slack" -> schedule_report

Question: {question}
Intent:"""


class IntentRouter:
    def __init__(self, config: IntentRouterConfig | None = None):
        self.config = config or IntentRouterConfig()
        self.llm = llm_service

    async def classify_intent(self, question: str) -> IntentClassification:
        prompt = INTENT_CLASSIFIER_PROMPT.format(question=question)

        try:
            result = await self._generate_structured(
                prompt=prompt,
                schema=IntentClassification,
                operation="intent_classification",
            )
            logger.info("intent_classified", intent=result.intent.value, confidence=result.confidence)
            return result
        except Exception as e:
            logger.warning("intent_classification_failed", error=str(e), question=question[:100])
            return IntentClassification(
                intent=QueryIntent.HYBRID,
                confidence=0.3,
                reasoning=f"Classification failed: {e}",
            )

    async def _generate_structured(
        self,
        prompt: str,
        schema: type[IntentClassification],
        operation: str,
    ) -> IntentClassification:
        format_prompt = f"""Return ONLY valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

No markdown, no explanation, no extra text."""

        full_prompt = f"{prompt}\n\n{format_prompt}"

        response = await self.llm.generate(
            full_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            format="json",
        )

        try:
            data = json.loads(response)
            return schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("structured_output_parse_failed", response=response, error=str(e))
            raise LLMError(f"Failed to parse structured output: {e}") from e

    async def classify_with_retry(
        self,
        question: str,
        max_retries: int = 3,
    ) -> IntentClassification:
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self.classify_intent(question)
            except Exception as e:
                last_error = e
                logger.warning("intent_classification_retry", attempt=attempt + 1, error=str(e))
                if attempt == max_retries - 1:
                    break
        raise LLMError(f"Intent classification failed after {max_retries} retries: {last_error}") from last_error