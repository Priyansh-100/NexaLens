import pytest
from uuid import uuid4, UUID

from nexalens.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_sql,
    evaluate_answer_faithfulness,
    evaluate_intent_classification,
    evaluate_tool_selection,
    RetrievalMetrics,
    SQLMetrics,
    AnswerFaithfulnessMetrics,
    IntentClassificationMetrics,
    ToolSelectionMetrics,
)
from nexalens.models.schemas import QueryIntent


class TestEvaluationMetrics:
    def test_evaluate_retrieval_perfect_match(self):
        retrieved = [
            {"id": uuid4()},
            {"id": uuid4()},
            {"id": uuid4()},
        ]
        relevant_ids = {retrieved[0]["id"], retrieved[1]["id"]}
        
        metrics = evaluate_retrieval(retrieved, relevant_ids, k_values=[1, 2, 3])
        
        assert metrics.hit_rate == 1.0
        assert metrics.recall_at_k[2] == 1.0
        assert metrics.precision_at_k[2] == 1.0
        assert metrics.mrr == 1.0

    def test_evaluate_retrieval_no_match(self):
        retrieved = [
            {"id": uuid4()},
            {"id": uuid4()},
        ]
        relevant_ids = {uuid4(), uuid4()}
        
        metrics = evaluate_retrieval(retrieved, relevant_ids, k_values=[1, 2])
        
        assert metrics.hit_rate == 0.0
        assert metrics.recall_at_k[2] == 0.0
        assert metrics.precision_at_k[2] == 0.0
        assert metrics.mrr == 0.0

    def test_evaluate_retrieval_partial_match(self):
        rel1 = uuid4()
        rel2 = uuid4()
        retrieved = [
            {"id": rel1},
            {"id": uuid4()},
            {"id": rel2},
        ]
        relevant_ids = {rel1, rel2}
        
        metrics = evaluate_retrieval(retrieved, relevant_ids, k_values=[1, 3])
        
        assert metrics.hit_rate == 1.0
        assert metrics.precision_at_k[1] == 1.0
        assert metrics.recall_at_k[1] == 0.5
        assert metrics.precision_at_k[3] == 2/3
        assert metrics.recall_at_k[3] == 1.0

    def test_evaluate_intent_classification(self):
        predictions = [
            QueryIntent.SQL,
            QueryIntent.DOCUMENT,
            QueryIntent.HYBRID,
            QueryIntent.FINANCIAL_MODEL,
            QueryIntent.FORECAST,
        ]
        ground_truth = [
            QueryIntent.SQL,
            QueryIntent.DOCUMENT,
            QueryIntent.SQL,  # Wrong
            QueryIntent.FINANCIAL_MODEL,
            QueryIntent.FORECAST,
        ]
        
        metrics = evaluate_intent_classification(predictions, ground_truth)
        
        assert metrics.accuracy == 0.8  # 4 out of 5 correct
        assert "sql" in metrics.per_class_f1
        assert "document" in metrics.per_class_f1

    def test_evaluate_tool_selection(self):
        predictions = [
            QueryIntent.SQL,
            QueryIntent.DOCUMENT,
            QueryIntent.HYBRID,
            QueryIntent.FINANCIAL_MODEL,
            QueryIntent.FORECAST,
        ]
        expected = [
            QueryIntent.SQL,
            QueryIntent.DOCUMENT,
            QueryIntent.HYBRID,
            QueryIntent.FINANCIAL_MODEL,
            QueryIntent.FORECAST,
        ]
        
        metrics = evaluate_tool_selection(predictions, expected)
        
        assert metrics.accuracy == 1.0
        assert metrics.sql_vs_rag_correct == 1.0
        assert metrics.hybrid_routing_correct == 1.0
        assert metrics.financial_model_correct == 1.0
        assert metrics.forecast_correct == 1.0

    def test_evaluate_answer_faithfulness(self):
        answer = "Revenue was $1M [source: sql, SELECT]. The report confirms growth [source: document, Q3 Report]."
        sql_result = MagicMock()
        doc_results = [{"document_id": uuid4(), "title": "Q3 Report"}]
        
        metrics = evaluate_answer_faithfulness(
            answer,
            sql_result,
            doc_results,
            None,
            None,
        )
        
        assert metrics.claims_total > 0
        assert metrics.citations_total == 2
        assert metrics.faithfulness_score > 0.5


class MockSQLResult:
    pass


class MockDocResult:
    pass