import time
from uuid import UUID

from nexalens.core.logging import get_logger
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
from nexalens.observability.tracing import get_correlation_id, trace_span
from nexalens.observability.metrics import record_llm_call, record_request_latency, record_sql_execution, record_retrieval
from nexalens.rag.services import (
    AnswerSynthesizer,
    FinancialAnalyzer,
    ForecastingServiceWrapper,
    IntentRouter,
    RAGRetriever,
    SQLAgent,
    ScheduleExtractor,
)

logger = get_logger(__name__)


class RAGOrchestrator:
    def __init__(self):
        self.intent_router = IntentRouter()
        self.sql_agent = SQLAgent()
        self.rag_retriever = RAGRetriever()
        self.financial_analyzer = FinancialAnalyzer()
        self.forecasting_service = ForecastingServiceWrapper()
        self.schedule_extractor = ScheduleExtractor()
        self.answer_synthesizer = AnswerSynthesizer()

    async def process_query(
        self,
        request: QueryRequest,
        session,
        user_id: UUID | None = None,
    ) -> QueryResponse:
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        logger.info(
            "query_started",
            question=request.question[:100],
            correlation_id=correlation_id,
        )

        intent_classification = await self.intent_router.classify_intent(request.question)
        intent = request.intent or intent_classification.intent

        logger.info("query_intent", question=request.question[:100], intent=intent.value, confidence=intent_classification.confidence)

        sql_result: SQLResult | None = None
        doc_results: list[DocumentResult] = []
        financial_result: FinancialModelResult | None = None
        forecast_result: ForecastResult | None = None
        schedule_result: ReportSchedule | None = None

        async with trace_span("intent_classification", intent=intent.value) as span:
            span.metadata["confidence"] = intent_classification.confidence

        if intent in (QueryIntent.SQL, QueryIntent.HYBRID):
            sql_result = await self._execute_sql(request, session)

        if intent in (QueryIntent.DOCUMENT, QueryIntent.HYBRID):
            doc_results = await self._search_documents(request)

        if intent == QueryIntent.FINANCIAL_MODEL:
            financial_result = await self.financial_analyzer.extract_and_run(request.question)

        if intent == QueryIntent.FORECAST:
            forecast_result = await self.forecasting_service.extract_and_forecast(
                request.question, session, request.data_source_ids or []
            )

        if intent == QueryIntent.SCHEDULE_REPORT:
            if user_id is None:
                raise ValueError("User ID required for scheduling reports")
            schedule_result = await self.schedule_extractor.extract_and_create(request.question, session, user_id)

        async with trace_span("answer_synthesis") as span:
            synthesis = await self.answer_synthesizer.synthesize(
                request.question,
                sql_result,
                doc_results,
                financial_result,
                forecast_result,
                schedule_result,
            )

        answer = synthesis.answer
        confidence = synthesis.confidence

        processing_time = (time.perf_counter() - start_time) * 1000
        record_request_latency(processing_time)

        logger.info(
            "query_completed",
            question=request.question[:100],
            intent=intent.value,
            confidence=confidence,
            processing_time_ms=processing_time,
            correlation_id=correlation_id,
        )

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

    async def _execute_sql(self, request: QueryRequest, session) -> SQLResult | None:
        try:
            data_source_ids = request.data_source_ids
            if not data_source_ids:
                return None

            ds_id = data_source_ids[0]

            from nexalens.models.database import DataSourceModel
            from sqlalchemy import select

            result = await session.execute(select(DataSourceModel).where(DataSourceModel.id == ds_id))
            data_source = result.scalar_one_or_none()
            if not data_source:
                logger.warning("datasource_not_found", datasource_id=str(ds_id))
                return None

            if data_source.type == "sql" and data_source.config.get("host"):
                return await self._execute_sql_on_external_datasource(request, ds_id, session)

            schema = await self.sql_agent.schema_service.get_schema(session, ds_id)
            allowed_tables = set(schema.keys())

            async with trace_span("sql_generation", datasource_id=str(ds_id)) as span:
                generation = await self.sql_agent.generate_sql(request.question, schema)
                span.metadata["tables_used"] = generation.tables_used

            async with trace_span("sql_execution", datasource_id=str(ds_id)) as span:
                result = await self.sql_agent.execute_sql(session, generation.sql, allowed_tables=allowed_tables)
                span.metadata["rows_returned"] = result.row_count
                span.metadata["execution_time_ms"] = result.execution_time_ms
                record_sql_execution(generation.sql, result.execution_time_ms, result.row_count)

            if request.include_sources and result.row_count > 0:
                explanation = await self.sql_agent.explain_sql_result(request.question, generation.sql, result)
                result.explanation = explanation

            return result
        except Exception as e:
            logger.error("sql_execution_failed", error=str(e))
            record_sql_execution("", 0, 0, status="error", error=str(e))
            return None

    async def _execute_sql_on_external_datasource(
        self,
        request: QueryRequest,
        ds_id: UUID,
        app_session,
    ) -> SQLResult | None:
        from nexalens.models.database import DataSourceModel
        from sqlalchemy import select
        from nexalens.sql.connection_manager import datasource_connection_manager

        try:
            result = await app_session.execute(select(DataSourceModel).where(DataSourceModel.id == ds_id))
            data_source = result.scalar_one_or_none()
            if not data_source:
                return None

            async with datasource_connection_manager.get_session(data_source) as ext_session:
                schema = await self.sql_agent.schema_service.get_schema(ext_session, ds_id)
                allowed_tables = set(schema.keys())

                async with trace_span("sql_generation", datasource_id=str(ds_id), external=True) as span:
                    generation = await self.sql_agent.generate_sql(request.question, schema)
                    span.metadata["tables_used"] = generation.tables_used

                async with trace_span("sql_execution", datasource_id=str(ds_id), external=True) as span:
                    result = await self.sql_agent.execute_sql(ext_session, generation.sql, allowed_tables=allowed_tables)
                    span.metadata["rows_returned"] = result.row_count
                    span.metadata["execution_time_ms"] = result.execution_time_ms
                    record_sql_execution(generation.sql, result.execution_time_ms, result.row_count)

                if request.include_sources and result.row_count > 0:
                    explanation = await self.sql_agent.explain_sql_result(request.question, generation.sql, result)
                    result.explanation = explanation

                return result
        except Exception as e:
            logger.error("external_sql_execution_failed", error=str(e), datasource_id=str(ds_id))
            return None

    async def _search_documents(self, request: QueryRequest) -> list[DocumentResult]:
        try:
            async with trace_span("document_retrieval", query=request.question[:100]) as span:
                results = await self.rag_retriever.search_documents(
                    request.question,
                    top_k=request.max_results,
                    source_ids=request.data_source_ids,
                )
                span.metadata["results_count"] = len(results)
                record_retrieval(request.question, len(results), span.duration_ms)

                doc_results = [
                    DocumentResult(
                        document_id=r.document_id,
                        title=r.title,
                        chunk_content=r.chunk_content,
                        score=r.score,
                        metadata=r.metadata,
                    )
                    for r in results
                ]
                return doc_results
        except Exception as e:
            logger.error("document_search_failed", error=str(e))
            return []


rag_orchestrator = RAGOrchestrator()