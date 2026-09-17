import json
import re
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.config import get_settings
from nexalens.core.exceptions import RetrievalError
from nexalens.core.logging import get_logger
from nexalens.models.database import DataSourceModel
from nexalens.models.schemas import ForecastModelType, ForecastRequest, ForecastResult
from nexalens.analytics.forecasting import forecasting_service
from nexalens.rag.schemas import LLMUsage, TimeSeriesIdentification
from nexalens.services.llm import llm_service
from nexalens.sql.connection_manager import datasource_connection_manager
from nexalens.sql.schema import schema_service
from nexalens.sql.service import sql_service

logger = get_logger(__name__)
settings = get_settings()

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ForecastingServiceWrapper:
    def __init__(self):
        self.llm = llm_service
        self.forecasting_service = forecasting_service
        self.schema_service = schema_service
        self.sql_service = sql_service

    async def extract_and_forecast(
        self,
        question: str,
        session: AsyncSession,
        data_source_ids: list[UUID],
    ) -> ForecastResult | None:
        try:
            if not data_source_ids:
                return None

            ds_id = data_source_ids[0]

            result = await session.execute(select(DataSourceModel).where(DataSourceModel.id == ds_id))
            data_source = result.scalar_one_or_none()
            if not data_source:
                logger.warning("datasource_not_found", datasource_id=str(ds_id))
                return None

            if data_source.type == "sql" and data_source.config.get("host"):
                async with datasource_connection_manager.get_session(data_source) as ext_session:
                    return await self._execute_forecast_on_session(question, ext_session, ds_id)
            else:
                return await self._execute_forecast_on_session(question, session, ds_id)

        except Exception as e:
            logger.error("forecast_failed", error=str(e), question=question[:100])
            return None

    async def _execute_forecast_on_session(
        self,
        question: str,
        ext_session: AsyncSession,
        ds_id: UUID,
    ) -> ForecastResult | None:
        try:
            schema = await self.schema_service.get_schema(ext_session, ds_id)

            table, metric_col, date_col = await self._identify_timeseries_table(schema, question)
            if not table:
                return None

            for name, field in [(table, "table"), (metric_col, "metric column"), (date_col, "date column")]:
                if not IDENTIFIER_RE.fullmatch(name):
                    logger.warning("invalid_identifier", identifier=name, field=field)
                    return None

            if table not in schema:
                logger.warning("table_not_in_schema", table=table)
                return None
            table_columns = {col["name"] for col in schema[table]}
            if metric_col not in table_columns or date_col not in table_columns:
                logger.warning("column_not_in_table", table=table, metric_col=metric_col, date_col=date_col)
                return None

            sql = f"SELECT {date_col}, {metric_col} FROM {table} ORDER BY {date_col}"
            sql_result = await self.sql_service.execute_sql(ext_session, sql)

            if sql_result.row_count < 3:
                logger.warning("insufficient_data_for_forecast", rows=sql_result.row_count)
                return None

            df = pd.DataFrame(sql_result.rows)

            fc_request = ForecastRequest(
                table_name=table,
                metric_column=metric_col,
                date_column=date_col,
                periods=12,
                frequency="M",
                model_type=ForecastModelType.PROPHET,
            )

            return self.forecasting_service.run_forecast(df, fc_request)

        except Exception as e:
            logger.error("forecast_on_session_failed", error=str(e))
            return None

    async def _identify_timeseries_table(
        self,
        schema: dict,
        question: str,
    ) -> tuple[str | None, str | None, str | None]:
        prompt = f"""Identify the best table and columns for time-series forecasting from this schema:
{schema}

Question: {question}

Return JSON with: table, metric_column, date_column
Example: {{"table": "orders", "metric_column": "amount", "date_column": "order_date"}}"""

        try:
            result = await self._generate_structured(
                prompt=prompt,
                schema=TimeSeriesIdentification,
                operation="timeseries_identification",
            )

            logger.debug(
                "timeseries_identified",
                table=result.table,
                metric_column=result.metric_column,
                date_column=result.date_column,
                confidence=result.confidence,
            )

            if result.confidence < 0.5:
                logger.warning("low_confidence_timeseries", confidence=result.confidence)
                return None, None, None

            return result.table, result.metric_column, result.date_column

        except Exception as e:
            logger.error("timeseries_identification_failed", error=str(e))
            return None, None, None

    async def _generate_structured(
        self,
        prompt: str,
        schema: type[TimeSeriesIdentification],
        operation: str,
    ) -> TimeSeriesIdentification:
        import json
        format_prompt = f"""Return ONLY valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

No markdown, no explanation, no extra text."""

        full_prompt = f"{prompt}\n\n{format_prompt}"

        response = await self.llm.generate(
            full_prompt,
            temperature=0.1,
            max_tokens=200,
            format="json",
        )

        try:
            data = json.loads(response)
            return schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("structured_timeseries_output_parse_failed", response=response, error=str(e))
            return TimeSeriesIdentification(
                table="",
                metric_column="",
                date_column="",
                confidence=0.0,
            )