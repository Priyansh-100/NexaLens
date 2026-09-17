import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nexalens.core.logging import get_logger
from nexalens.evaluation.metrics import (
    EvaluationResult,
    RetrievalMetrics,
    SQLMetrics,
    AnswerFaithfulnessMetrics,
    IntentClassificationMetrics,
    ToolSelectionMetrics,
    evaluate_retrieval,
    evaluate_sql,
    evaluate_answer_faithfulness,
    evaluate_intent_classification,
    evaluate_tool_selection,
)
from nexalens.models.database import QueryLogModel
from nexalens.models.schemas import QueryIntent
from nexalens.models.session import get_db_session
from nexalens.rag.orchestrator import rag_orchestrator
from nexalens.services.embeddings import embedding_service

logger = get_logger(__name__)


class EvaluationRunner:
    def __init__(self, dataset_path: Path | None = None):
        self.dataset_path = dataset_path or Path("tests/eval_dataset.json")
        self.results: list[EvaluationResult] = []
        self.all_predictions: list[QueryIntent] = []
        self.all_ground_truth: list[QueryIntent] = []

    async def run(self, session: AsyncSession, limit: int | None = None) -> list[EvaluationResult]:
        dataset = self._load_dataset()
        if limit:
            dataset = dataset[:limit]

        logger.info("evaluation_started", total=len(dataset))

        for item in dataset:
            result = await self._evaluate_item(session, item)
            self.results.append(result)

            if item.get("intent"):
                self.all_predictions.append(result.intent)
                self.all_ground_truth.append(QueryIntent(item["intent"]))

        self._evaluate_intent_and_tools()
        self._print_summary()
        return self.results

    def _load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            logger.warning("eval_dataset_not_found", path=str(self.dataset_path))
            return []

        with open(self.dataset_path) as f:
            return json.load(f)

    async def _evaluate_item(self, session: AsyncSession, item: dict[str, Any]) -> EvaluationResult:
        query_id = UUID(item.get("id", str(uuid4())))
        question = item["question"]
        expected_intent = item.get("intent")
        relevant_doc_ids = {UUID(rid) for rid in item.get("relevant_doc_ids", [])}
        expected_sql = item.get("expected_sql")

        start_time = time.perf_counter()

        from nexalens.models.schemas import QueryRequest
        request = QueryRequest(
            question=question,
            intent=QueryIntent(expected_intent) if expected_intent else None,
            max_results=10,
            include_sql=True,
            include_sources=True,
        )

        response = await rag_orchestrator.process_query(request, session)

        latency_ms = (time.perf_counter() - start_time) * 1000

        retrieval_metrics = None
        if relevant_doc_ids and response.document_results:
            retrieved = [{"id": r.document_id} for r in response.document_results]
            retrieval_metrics = evaluate_retrieval(retrieved, relevant_doc_ids)

        sql_metrics = None
        if expected_sql and response.sql_result:
            sql_metrics = evaluate_sql(response.sql_result.sql, expected_sql, session)

        faithfulness_metrics = evaluate_answer_faithfulness(
            response.answer,
            response.sql_result,
            [{"document_id": r.document_id, "title": r.title} for r in response.document_results],
            response.financial_result,
            response.forecast_result,
        )

        passed = True
        if retrieval_metrics and retrieval_metrics.hit_rate < 0.5:
            passed = False
        if sql_metrics and sql_metrics.execution_accuracy < 1.0:
            passed = False
        if faithfulness_metrics.faithfulness_score < 0.7:
            passed = False

        return EvaluationResult(
            query_id=query_id,
            question=question,
            intent=response.intent.value,
            retrieval=retrieval_metrics,
            sql=sql_metrics,
            faithfulness=faithfulness_metrics,
            latency_ms=latency_ms,
            confidence=response.confidence,
            passed=passed,
        )

    def _evaluate_intent_and_tools(self) -> None:
        if not self.all_predictions or not self.all_ground_truth:
            return

        intent_metrics = evaluate_intent_classification(
            self.all_predictions, self.all_ground_truth
        )
        tool_metrics = evaluate_tool_selection(
            self.all_predictions, self.all_ground_truth
        )

        for result in self.results:
            result.intent_metrics = intent_metrics
            result.tool_selection = tool_metrics

    def _print_summary(self) -> None:
        if not self.results:
            return

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        avg_latency = sum(r.latency_ms for r in self.results) / total
        avg_confidence = sum(r.confidence for r in self.results) / total

        retrieval_hits = [r.retrieval.hit_rate for r in self.results if r.retrieval]
        sql_acc = [r.sql.execution_accuracy for r in self.results if r.sql]
        faithfulness_scores = [r.faithfulness.faithfulness_score for r in self.results if r.faithfulness]

        logger.info(
            "evaluation_complete",
            total=total,
            passed=passed,
            pass_rate=f"{passed/total:.1%}",
            avg_latency_ms=f"{avg_latency:.0f}",
            avg_confidence=f"{avg_confidence:.1%}",
            retrieval_hit_rate=f"{sum(retrieval_hits)/len(retrieval_hits):.1%}" if retrieval_hits else "N/A",
            sql_execution_accuracy=f"{sum(sql_acc)/len(sql_acc):.1%}" if sql_acc else "N/A",
            answer_faithfulness=f"{sum(faithfulness_scores)/len(faithfulness_scores):.1%}" if faithfulness_scores else "N/A",
        )

        if self.results[0].intent_metrics:
            im = self.results[0].intent_metrics
            logger.info(
                "intent_classification_metrics",
                accuracy=f"{im.accuracy:.1%}",
                macro_f1=f"{sum(im.per_class_f1.values())/len(im.per_class_f1):.1%}" if im.per_class_f1 else "N/A",
            )

        if self.results[0].tool_selection:
            tm = self.results[0].tool_selection
            logger.info(
                "tool_selection_metrics",
                accuracy=f"{tm.accuracy:.1%}",
                sql_vs_rag=f"{tm.sql_vs_rag_correct:.1%}",
                hybrid=f"{tm.hybrid_routing_correct:.1%}",
                financial=f"{tm.financial_model_correct:.1%}",
                forecast=f"{tm.forecast_correct:.1%}",
                schedule=f"{tm.schedule_correct:.1%}",
            )

    def export_results(self, output_path: Path) -> None:
        data = [asdict(r) for r in self.results]
        for d in data:
            d["query_id"] = str(d["query_id"])
            if d.get("retrieval"):
                d["retrieval"] = asdict(d["retrieval"])
            if d.get("sql"):
                d["sql"] = asdict(d["sql"])
            if d.get("faithfulness"):
                d["faithfulness"] = asdict(d["faithfulness"])
            if d.get("intent_metrics"):
                d["intent_metrics"] = asdict(d["intent_metrics"])
            if d.get("tool_selection"):
                d["tool_selection"] = asdict(d["tool_selection"])

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("results_exported", path=str(output_path))


async def run_evaluation(limit: int | None = None) -> list[EvaluationResult]:
    async for session in get_db_session():
        runner = EvaluationRunner()
        return await runner.run(session, limit)