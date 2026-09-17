import json
from typing import Any
from uuid import UUID

from nexalens.core.config import get_settings
from nexalens.core.exceptions import LLMError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import ReportFormat, ReportSchedule, ReportScheduleCreate
from nexalens.analytics.scheduler import report_scheduler
from nexalens.rag.schemas import LLMUsage, ScheduleExtraction
from nexalens.services.llm import llm_service

logger = get_logger(__name__)
settings = get_settings()


class ScheduleExtractor:
    def __init__(self):
        self.llm = llm_service
        self.scheduler_service = report_scheduler

    async def extract_and_create(
        self,
        question: str,
        session,
        user_id: UUID,
    ) -> ReportSchedule | None:
        try:
            extraction = await self._extract_schedule_params(question)

            schedule_in = ReportScheduleCreate(
                name=extraction.name,
                query=extraction.query,
                data_source_ids=extraction.data_source_ids,
                cron_expression=extraction.cron_expression,
                recipients=extraction.recipients,
                format=extraction.format,
                template=extraction.template,
            )

            result = await self.scheduler_service.create_schedule(schedule_in, session, user_id)
            logger.info("schedule_created", name=result.name, schedule_id=str(result.id))
            return result

        except Exception as e:
            logger.error("schedule_creation_failed", error=str(e), question=question[:100])
            return None

    async def _extract_schedule_params(self, question: str) -> ScheduleExtraction:
        prompt = f"""Extract report schedule parameters from this question:
{question}

Return JSON with:
- name: string
- query: string (the question to run on schedule)
- data_source_ids: array of UUID strings
- cron_expression: string (standard cron format)
- recipients: array of email strings
- format: one of [pdf, excel, csv, html]
- template: string or null
- confidence: 0.0 to 1.0

Return null if not a scheduling request."""

        result = await self._generate_structured(
            prompt=prompt,
            schema=ScheduleExtraction,
            operation="schedule_extraction",
        )

        logger.debug(
            "schedule_extracted",
            name=result.name,
            cron=result.cron_expression,
            recipients=result.recipients,
        )
        return result

    async def _generate_structured(
        self,
        prompt: str,
        schema: type[ScheduleExtraction],
        operation: str,
    ) -> ScheduleExtraction:
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
            if "data_source_ids" in data:
                data["data_source_ids"] = [UUID(s) for s in data["data_source_ids"]]
            return schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("structured_schedule_output_parse_failed", response=response, error=str(e))
            raise LLMError(f"Failed to parse structured schedule output: {e}") from e