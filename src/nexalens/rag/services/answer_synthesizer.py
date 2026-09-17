import json
from typing import Any
from uuid import UUID

from nexalens.core.config import get_settings
from nexalens.core.exceptions import LLMError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import (
    DocumentResult,
    FinancialModelResult,
    ForecastResult,
    ReportSchedule,
    SQLResult,
)
from nexalens.rag.quality import CitationTracker, ConfidenceEvaluator, rag_quality_config
from nexalens.rag.schemas import AnswerSynthesis, AnswerSynthesizerConfig, LLMUsage
from nexalens.services.llm import llm_service

logger = get_logger(__name__)
settings = get_settings()

ANSWER_SYNTHESIS_PROMPT = """You are a business analytics assistant. Synthesize a clear, actionable answer from the retrieved data.

Question: {question}

SQL Results:
{sql_results}

Document Results:
{doc_results}

Financial Model Result:
{financial_result}

Forecast Result:
{forecast_result}

Schedule Result:
{schedule_result}

Guidelines:
1. Be concise but thorough
2. Cite specific numbers from SQL results
3. Reference document sources when relevant
4. Highlight key insights and anomalies
5. If data is insufficient, say so
6. Format with clear sections for readability
7. Include citations in format [source: type, id] where type is sql/document/financial/forecast

Answer:"""


class AnswerSynthesizer:
    def __init__(self, config: AnswerSynthesizerConfig | None = None):
        self.config = config or AnswerSynthesizerConfig()
        self.llm = llm_service

        self.quality_config = rag_quality_config
        self.citation_tracker = CitationTracker(
            max_citations=self.quality_config.max_citations_per_answer,
            format=self.quality_config.citation_format,
        )
        self.confidence_evaluator = ConfidenceEvaluator(
            threshold=self.quality_config.confidence_threshold,
            unknown_response=self.quality_config.unknown_response,
        )

    async def synthesize(
        self,
        question: str,
        sql_result: SQLResult | None,
        doc_results: list[DocumentResult],
        financial_result: FinancialModelResult | None = None,
        forecast_result: ForecastResult | None = None,
        schedule_result: ReportSchedule | None = None,
    ) -> AnswerSynthesis:
        self.citation_tracker.reset()
        parts = []

        if financial_result:
            parts.append(f"Financial Model ({financial_result.model_type.value.upper()}):\n{financial_result.explanation}")
            if financial_result.npv is not None:
                parts.append(f"NPV: ${financial_result.npv:,.2f}")
            if financial_result.irr is not None:
                parts.append(f"IRR: {financial_result.irr:.2%}")
            if financial_result.dcf_value is not None:
                parts.append(f"DCF Value: ${financial_result.dcf_value:,.2f}")
            if financial_result.payback_period is not None:
                parts.append(f"Payback: {financial_result.payback_period:.2f} years")
            if financial_result.excel_base64:
                parts.append("[Excel file generated - available for download]")
            self.citation_tracker.add_citation("financial", str(financial_result.model_type.value))

        if forecast_result:
            parts.append(f"Forecast ({forecast_result.model_type.upper()}):\n{forecast_result.explanation}")
            if forecast_result.forecast:
                next_fc = forecast_result.forecast[0]
                parts.append(
                    f"Next period: {next_fc.date} = {next_fc.yhat:,.2f} "
                    f"(CI: {next_fc.yhat_lower:,.2f} - {next_fc.yhat_upper:,.2f})"
                )
            self.citation_tracker.add_citation("forecast", forecast_result.model_type)

        if schedule_result:
            parts.append(f"Scheduled report created: '{schedule_result.name}'")
            parts.append(f"Cron: {schedule_result.cron_expression}")
            parts.append(f"Recipients: {', '.join(schedule_result.recipients)}")
            parts.append(f"Format: {schedule_result.format.value}")
            self.citation_tracker.add_citation("schedule", str(schedule_result.id))

        sql_text = "No SQL results."
        if sql_result:
            sql_text = f"Query returned {sql_result.row_count} rows.\n"
            sql_text += f"Columns: {', '.join(sql_result.columns)}\n"
            if sql_result.rows:
                sql_text += f"Sample data: {sql_result.rows[:3]}\n"
            if sql_result.explanation:
                sql_text += f"Explanation: {sql_result.explanation}"
            self.citation_tracker.add_citation("sql", sql_result.sql[:50])

        doc_text = "No document results."
        if doc_results:
            doc_text = f"Found {len(doc_results)} relevant document chunks:\n"
            for i, doc in enumerate(doc_results[:3]):
                doc_text += f"{i+1}. [{doc.title}] (score: {doc.score:.2f}): {doc.chunk_content[:200]}...\n"
                self.citation_tracker.add_citation("document", str(doc.document_id), title=doc.title)

        if parts:
            answer = "\n\n".join(parts)
            citations = self.citation_tracker.get_citations()
            confidence, unknown_response = self.confidence_evaluator.evaluate(
                sql_result,
                [{"score": d.score, "metadata": d.metadata} for d in doc_results],
                financial_result,
                forecast_result,
            )
            return AnswerSynthesis(
                answer=answer,
                confidence=confidence,
                citations=citations,
                data_sufficiency="sufficient" if not unknown_response else "insufficient",
            )

        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            sql_results=sql_text,
            doc_results=doc_text,
            financial_result="N/A" if not financial_result else "See above",
            forecast_result="N/A" if not forecast_result else "See above",
            schedule_result="N/A" if not schedule_result else "See above",
        )

        try:
            result = await self._generate_structured(
                prompt=prompt,
                schema=AnswerSynthesis,
                operation="answer_synthesis",
            )

            citations = self.citation_tracker.get_citations()
            result.citations = citations

            confidence, unknown_response = self.confidence_evaluator.evaluate(
                sql_result,
                [{"score": d.score, "metadata": d.metadata} for d in doc_results],
                financial_result,
                forecast_result,
            )
            result.confidence = confidence

            if unknown_response:
                result.answer = unknown_response
                result.data_sufficiency = "insufficient"

            return result

        except Exception as e:
            logger.error("answer_synthesis_failed", error=str(e))
            confidence, unknown_response = self.confidence_evaluator.evaluate(
                sql_result,
                [{"score": d.score, "metadata": d.metadata} for d in doc_results],
                financial_result,
                forecast_result,
            )
            return AnswerSynthesis(
                answer=unknown_response or "I found relevant data but couldn't synthesize a complete answer. Please check the raw results.",
                confidence=confidence,
                citations=self.citation_tracker.get_citations(),
                data_sufficiency="partial" if not unknown_response else "insufficient",
            )

    async def _generate_structured(
        self,
        prompt: str,
        schema: type[AnswerSynthesis],
        operation: str,
    ) -> AnswerSynthesis:
        import json
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
            logger.error("structured_answer_output_parse_failed", response=response, error=str(e))
            raise LLMError(f"Failed to parse structured answer output: {e}") from e