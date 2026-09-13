import time
from typing import Any
from uuid import UUID

from nexalens.core.logging import get_logger
from nexalens.core.exceptions import RetrievalError
from nexalens.models.schemas import (
    DocumentResult,
    FinancialModelResult,
    ForecastResult,
    QueryIntent,
    QueryRequest,
    QueryResponse,
    ReportSchedule,
    SQLResult,
)
from nexalens.services.embeddings import embedding_service
from nexalens.services.llm import llm_service
from nexalens.sql.schema import schema_service
from nexalens.sql.service import sql_service
from nexalens.sql.connection_manager import datasource_connection_manager
from nexalens.documents.service import document_service
from nexalens.analytics.financial import financial_modeling_service
from nexalens.analytics.forecasting import forecasting_service
from nexalens.analytics.scheduler import report_scheduler
from nexalens.models.database import DataSourceModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


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

ANSWER_SYNTHESIS_PROMPT = """You are a business analytics assistant. Synthesize a clear, actionable answer from the retrieved data.

Question: {question}

SQL Results:
{sql_results}

Document Results:
{doc_results}

Guidelines:
1. Be concise but thorough
2. Cite specific numbers from SQL results
3. Reference document sources when relevant
4. Highlight key insights and anomalies
5. If data is insufficient, say so
6. Format with clear sections for readability

Answer:"""


class RAGOrchestrator:
    def __init__(self):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.sql_service = sql_service
        self.schema_service = schema_service
        self.document_service = document_service
        self.financial_service = financial_modeling_service
        self.forecasting_service = forecasting_service
        self.scheduler_service = report_scheduler

    async def classify_intent(self, question: str) -> QueryIntent:
        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(question=question)
            response = await self.llm_service.generate(prompt, temperature=0.0, max_tokens=50)
            intent_str = response.strip().lower()

            for intent in QueryIntent:
                if intent.value in intent_str:
                    return intent
            return QueryIntent.CLARIFICATION
        except Exception as e:
            logger.warning("intent_classification_failed", error=str(e))
            return QueryIntent.HYBRID

    async def process_query(self, request: QueryRequest, session, user_id: UUID | None = None) -> QueryResponse:
        start_time = time.perf_counter()

        intent = request.intent or await self.classify_intent(request.question)
        logger.info("query_intent", question=request.question[:100], intent=intent.value)

        sql_result: SQLResult | None = None
        doc_results: list[DocumentResult] = []
        financial_result: FinancialModelResult | None = None
        forecast_result: ForecastResult | None = None
        schedule_result: ReportSchedule | None = None

        if intent in (QueryIntent.SQL, QueryIntent.HYBRID):
            sql_result = await self._execute_sql(request, session)

        if intent in (QueryIntent.DOCUMENT, QueryIntent.HYBRID):
            doc_results = await self._search_documents(request)

        if intent == QueryIntent.FINANCIAL_MODEL:
            financial_result = await self._execute_financial_model(request, session)

        if intent == QueryIntent.FORECAST:
            forecast_result = await self._execute_forecast(request, session)

        if intent == QueryIntent.SCHEDULE_REPORT:
            if user_id is None:
                raise ValueError("User ID required for scheduling reports")
            schedule_result = await self._create_schedule(request, session, user_id)

        answer = await self._synthesize_answer(request.question, sql_result, doc_results, financial_result, forecast_result, schedule_result)
        confidence = self._calculate_confidence(intent, sql_result, doc_results, financial_result, forecast_result)

        processing_time = (time.perf_counter() - start_time) * 1000

        return QueryResponse(
            question=request.question,
            intent=intent,
            answer=answer,
            sql_result=sql_result,
            document_results=doc_results,
            financial_result=financial_result,
            forecast_result=forecast_result,
            schedule_result=schedule_result,
            confidence=confidence,
            processing_time_ms=processing_time,
        )

    async def _execute_sql(self, request: QueryRequest, session: AsyncSession) -> SQLResult | None:
        try:
            data_source_ids = request.data_source_ids
            if not data_source_ids:
                return None

            ds_id = data_source_ids[0]

            # Get the data source model to check type and config
            result = await session.execute(select(DataSourceModel).where(DataSourceModel.id == ds_id))
            data_source = result.scalar_one_or_none()
            if not data_source:
                logger.warning("datasource_not_found", datasource_id=str(ds_id))
                return None

            # If it's an external SQL data source, use connection manager
            if data_source.type == "sql" and data_source.config.get("host"):
                return await self._execute_sql_on_external_datasource(request, ds_id, session)

            # Otherwise use the application database (for internal/metadata queries)
            schema = await self.schema_service.get_schema(session, ds_id)
            allowed_tables = set(schema.keys())
            sql = await self.sql_service.generate_sql(request.question, schema)
            logger.debug("generated_sql", sql=sql[:200])

            result = await self.sql_service.execute_sql(session, sql, allowed_tables=allowed_tables)

            if request.include_sources and result.row_count > 0:
                explanation = await self._explain_sql_result(request.question, sql, result)
                result.explanation = explanation

            return result
        except Exception as e:
            logger.error("sql_execution_failed", error=str(e))
            return None

    async def _execute_sql_on_external_datasource(
        self,
        request: QueryRequest,
        ds_id: UUID,
        app_session: AsyncSession,
    ) -> SQLResult | None:
        """Execute SQL on an external data source using connection manager."""
        try:
            # Get the data source again from app session for config
            result = await app_session.execute(select(DataSourceModel).where(DataSourceModel.id == ds_id))
            data_source = result.scalar_one_or_none()
            if not data_source:
                return None

            # Use connection manager to get a session on the external DB
            async with datasource_connection_manager.get_session(data_source) as ext_session:
                schema = await self.schema_service.get_schema(ext_session, ds_id)
                allowed_tables = set(schema.keys())
                sql = await self.sql_service.generate_sql(request.question, schema)
                logger.debug("generated_sql_external", sql=sql[:200])

                result = await self.sql_service.execute_sql(ext_session, sql, allowed_tables=allowed_tables)

                if request.include_sources and result.row_count > 0:
                    explanation = await self._explain_sql_result(request.question, sql, result)
                    result.explanation = explanation

                return result
        except Exception as e:
            logger.error("external_sql_execution_failed", error=str(e), datasource_id=str(ds_id))
            return None

    async def _search_documents(self, request: QueryRequest) -> list[DocumentResult]:
        try:
            query_embedding = await self.embedding_service.embed(request.question)
            chunks = await self.document_service.search_documents(
                query_embedding,
                top_k=request.max_results,
                source_ids=request.data_source_ids,
            )

            return [
                DocumentResult(
                    document_id=chunk["metadata"]["document_id"],
                    title=chunk["metadata"].get("document_title", "Unknown"),
                    chunk_content=chunk["content"][:500],
                    score=chunk["score"],
                    metadata=chunk["metadata"],
                )
                for chunk in chunks
            ]
        except Exception as e:
            logger.error("document_search_failed", error=str(e))
            return []

    async def _explain_sql_result(self, question: str, sql: str, result: SQLResult) -> str:
        prompt = f"""Explain this SQL result in business terms for: "{question}"

SQL: {sql}
Results: {result.row_count} rows, columns: {result.columns}
Sample: {result.rows[:3]}

Provide a 2-3 sentence business interpretation."""
        try:
            return await self.llm_service.generate(prompt, temperature=0.1, max_tokens=300)
        except Exception:
            return "Query executed successfully."

    async def _execute_financial_model(self, request: QueryRequest, session) -> FinancialModelResult | None:
        try:
            from nexalens.models.schemas import FinancialModelRequest, FinancialModelType

            model_type = await self._extract_model_type(request.question)
            assumptions = await self._extract_assumptions(request.question, model_type)
            cash_flows = await self._extract_cash_flows(request.question)

            fm_request = FinancialModelRequest(
                model_type=model_type,
                assumptions=assumptions,
                cash_flows=cash_flows,
                output_format="excel",
            )

            return self.financial_service.run_model(fm_request)
        except Exception as e:
            logger.error("financial_model_failed", error=str(e))
            return None

    async def _execute_forecast(self, request: QueryRequest, session: AsyncSession) -> ForecastResult | None:
        try:
            from nexalens.models.schemas import ForecastRequest, ForecastModelType
            import pandas as pd
            import re

            data_source_ids = request.data_source_ids
            if not data_source_ids:
                return None

            ds_id = data_source_ids[0]

            # Get the data source model
            result = await session.execute(select(DataSourceModel).where(DataSourceModel.id == ds_id))
            data_source = result.scalar_one_or_none()
            if not data_source:
                return None

            # Determine which session to use
            if data_source.type == "sql" and data_source.config.get("host"):
                # External data source - use connection manager
                async with datasource_connection_manager.get_session(data_source) as ext_session:
                    return await self._execute_forecast_on_session(request, ext_session, ds_id)
            else:
                # Use application database
                return await self._execute_forecast_on_session(request, session, ds_id)
        except Exception as e:
            logger.error("forecast_failed", error=str(e))
            return None

    async def _execute_forecast_on_session(
        self,
        request: QueryRequest,
        ext_session: AsyncSession,
        ds_id: UUID,
    ) -> ForecastResult | None:
        """Execute forecast logic on a given session (external or app)."""
        try:
            from nexalens.models.schemas import ForecastRequest, ForecastModelType
            import pandas as pd
            import re

            schema = await self.schema_service.get_schema(ext_session, ds_id)

            table, metric_col, date_col = await self._identify_timeseries_table(schema, request.question)
            if not table:
                return None

            # Validate identifiers
            identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
            for name, field in [(table, "table"), (metric_col, "metric column"), (date_col, "date column")]:
                if not identifier_re.fullmatch(name):
                    logger.warning("invalid_identifier", identifier=name, field=field)
                    return None

            # Verify table and columns exist in schema
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

    async def _create_schedule(self, request: QueryRequest, session, user_id: UUID) -> ReportSchedule | None:
        try:
            from nexalens.models.schemas import ReportScheduleCreate, ReportFormat

            schedule_data = await self._extract_schedule_params(request.question)
            if not schedule_data:
                return None

            schedule_in = ReportScheduleCreate(**schedule_data)
            return await self.scheduler_service.create_schedule(schedule_in, session, user_id)
        except Exception as e:
            logger.error("schedule_creation_failed", error=str(e))
            return None

    async def _extract_model_type(self, question: str) -> "FinancialModelType":
        from nexalens.models.schemas import FinancialModelType

        prompt = f"""Extract the financial model type from this question:
{question}

Options: dcf, npv, irr, payback, sensitivity
Return only the model type (one word)."""
        response = await self.llm_service.generate(prompt, temperature=0.0, max_tokens=20)
        try:
            return FinancialModelType(response.strip().lower())
        except Exception:
            return FinancialModelType.DCF

    async def _extract_assumptions(self, question: str, model_type: "FinancialModelType") -> dict[str, Any]:
        prompt = f"""Extract financial assumptions from this question as JSON:
{question}

Model type: {model_type.value}
Return JSON with keys like: discount_rate, terminal_growth, revenue_growth, tax_rate, etc.
Only include values explicitly mentioned or reasonable defaults."""
        response = await self.llm_service.generate(prompt, temperature=0.1, max_tokens=300)
        try:
            import json
            return json.loads(response)
        except Exception:
            return {}

    async def _extract_cash_flows(self, question: str) -> list[float] | None:
        prompt = f"""Extract cash flows array from this question:
{question}

Return as JSON array of numbers, e.g. [-100000, 30000, 40000, 50000]
Return null if not found."""
        response = await self.llm_service.generate(prompt, temperature=0.0, max_tokens=100)
        try:
            import json
            result = json.loads(response)
            return result if isinstance(result, list) else None
        except Exception:
            return None

    async def _identify_timeseries_table(self, schema: dict, question: str) -> tuple[str | None, str | None, str | None]:
        prompt = f"""Identify the best table and columns for time-series forecasting from this schema:
{schema}

Question: {question}

Return JSON with: table, metric_column, date_column
Example: {{"table": "orders", "metric_column": "amount", "date_column": "order_date"}}"""
        response = await self.llm_service.generate(prompt, temperature=0.1, max_tokens=200)
        try:
            import json
            result = json.loads(response)
            return result.get("table"), result.get("metric_column"), result.get("date_column")
        except Exception:
            return None, None, None

    async def _extract_schedule_params(self, question: str) -> dict[str, Any] | None:
        prompt = f"""Extract report schedule parameters from this question:
{question}

Return JSON with: name, query, data_source_ids (array of UUIDs), cron_expression, recipients (array), format (pdf/excel/csv/html)
Return null if not a scheduling request."""
        response = await self.llm_service.generate(prompt, temperature=0.1, max_tokens=300)
        try:
            import json
            result = json.loads(response)
            if "data_source_ids" in result:
                from uuid import UUID
                result["data_source_ids"] = [UUID(s) for s in result["data_source_ids"]]
            return result
        except Exception:
            return None

    async def _synthesize_answer(
        self,
        question: str,
        sql_result: SQLResult | None,
        doc_results: list[DocumentResult],
        financial_result: FinancialModelResult | None = None,
        forecast_result: ForecastResult | None = None,
        schedule_result: ReportSchedule | None = None,
    ) -> str:
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

        if forecast_result:
            parts.append(f"Forecast ({forecast_result.model_type.upper()}):\n{forecast_result.explanation}")
            if forecast_result.forecast:
                next_fc = forecast_result.forecast[0]
                parts.append(f"Next period: {next_fc.date} = {next_fc.yhat:,.2f} (CI: {next_fc.yhat_lower:,.2f} - {next_fc.yhat_upper:,.2f})")

        if schedule_result:
            parts.append(f"Scheduled report created: '{schedule_result.name}'")
            parts.append(f"Cron: {schedule_result.cron_expression}")
            parts.append(f"Recipients: {', '.join(schedule_result.recipients)}")
            parts.append(f"Format: {schedule_result.format.value}")

        sql_text = "No SQL results."
        if sql_result:
            sql_text = f"Query returned {sql_result.row_count} rows.\n"
            sql_text += f"Columns: {', '.join(sql_result.columns)}\n"
            if sql_result.rows:
                sql_text += f"Sample data: {sql_result.rows[:3]}\n"
            if sql_result.explanation:
                sql_text += f"Explanation: {sql_result.explanation}"

        doc_text = "No document results."
        if doc_results:
            doc_text = f"Found {len(doc_results)} relevant document chunks:\n"
            for i, doc in enumerate(doc_results[:3]):
                doc_text += f"{i+1}. [{doc.title}] (score: {doc.score:.2f}): {doc.chunk_content[:200]}...\n"

        if parts:
            return "\n\n".join(parts)

        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            sql_results=sql_text,
            doc_results=doc_text,
        )

        try:
            return await self.llm_service.generate(prompt, temperature=0.2, max_tokens=1024)
        except Exception as e:
            logger.error("answer_synthesis_failed", error=str(e))
            return "I found relevant data but couldn't synthesize a complete answer. Please check the raw results."

    def _calculate_confidence(
        self,
        intent: QueryIntent,
        sql_result: SQLResult | None,
        doc_results: list[DocumentResult],
        financial_result: FinancialModelResult | None = None,
        forecast_result: ForecastResult | None = None,
    ) -> float:
        scores = []

        if intent in (QueryIntent.SQL, QueryIntent.HYBRID) and sql_result:
            scores.append(min(1.0, sql_result.row_count / 10.0) * 0.8 + 0.2)

        if intent in (QueryIntent.DOCUMENT, QueryIntent.HYBRID) and doc_results:
            avg_score = sum(d.score for d in doc_results) / len(doc_results)
            scores.append(avg_score)

        if intent == QueryIntent.FINANCIAL_MODEL and financial_result:
            scores.append(0.9)

        if intent == QueryIntent.FORECAST and forecast_result:
            scores.append(0.85)

        if intent == QueryIntent.SCHEDULE_REPORT:
            scores.append(0.95)

        if intent == QueryIntent.CLARIFICATION:
            return 0.3

        return sum(scores) / len(scores) if scores else 0.5


rag_orchestrator = RAGOrchestrator()