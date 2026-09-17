from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from nexalens.models.schemas import QueryIntent


@dataclass
class RetrievalMetrics:
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0


@dataclass
class SQLMetrics:
    execution_accuracy: float = 0.0
    exact_match_accuracy: float = 0.0
    valid_sql_rate: float = 0.0
    avg_execution_time_ms: float = 0.0


@dataclass
class AnswerFaithfulnessMetrics:
    faithfulness_score: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    hallucination_rate: float = 0.0
    claims_total: int = 0
    claims_supported: int = 0
    citations_total: int = 0
    citations_valid: int = 0


@dataclass
class IntentClassificationMetrics:
    accuracy: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ToolSelectionMetrics:
    accuracy: float = 0.0
    sql_vs_rag_correct: float = 0.0
    hybrid_routing_correct: float = 0.0
    financial_model_correct: float = 0.0
    forecast_correct: float = 0.0
    schedule_correct: float = 0.0


@dataclass
class EvaluationResult:
    query_id: UUID
    question: str
    intent: str
    retrieval: RetrievalMetrics | None = None
    sql: SQLMetrics | None = None
    faithfulness: AnswerFaithfulnessMetrics | None = None
    intent_metrics: IntentClassificationMetrics | None = None
    tool_selection: ToolSelectionMetrics | None = None
    latency_ms: float = 0.0
    confidence: float = 0.0
    passed: bool = False


def evaluate_retrieval(
    retrieved: list[dict[str, Any]],
    relevant_ids: set[UUID],
    k_values: list[int] = None,
) -> RetrievalMetrics:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    retrieved_ids = [item["id"] for item in retrieved]
    metrics = RetrievalMetrics()

    for k in k_values:
        top_k = retrieved_ids[:k]
        hits = sum(1 for rid in top_k if rid in relevant_ids)

        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant_ids) if relevant_ids else 0.0

        metrics.precision_at_k[k] = precision
        metrics.recall_at_k[k] = recall

        dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(top_k) if rid in relevant_ids)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics.ndcg_at_k[k] = ndcg

    reciprocal_ranks = []
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            reciprocal_ranks.append(1.0 / (i + 1))
            break
    metrics.mrr = reciprocal_ranks[0] if reciprocal_ranks else 0.0

    metrics.hit_rate = 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0

    return metrics


def evaluate_sql(
    predicted_sql: str,
    expected_sql: str,
    session,
    execution_timeout: float = 30.0,
) -> SQLMetrics:
    import time
    from sqlalchemy import text

    metrics = SQLMetrics()
    metrics.valid_sql_rate = 1.0

    pred_normalized = " ".join(predicted_sql.strip().split()).lower()
    exp_normalized = " ".join(expected_sql.strip().split()).lower()
    metrics.exact_match_accuracy = 1.0 if pred_normalized == exp_normalized else 0.0

    try:
        start = time.perf_counter()
        pred_result = session.execute(text(predicted_sql)).fetchall()
        pred_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        exp_result = session.execute(text(expected_sql)).fetchall()
        exp_time = (time.perf_counter() - start) * 1000

        metrics.avg_execution_time_ms = (pred_time + exp_time) / 2

        pred_set = set(str(row) for row in pred_result)
        exp_set = set(str(row) for row in exp_result)
        metrics.execution_accuracy = 1.0 if pred_set == exp_set else 0.0

    except Exception:
        metrics.valid_sql_rate = 0.0
        metrics.execution_accuracy = 0.0

    return metrics


def evaluate_answer_faithfulness(
    answer: str,
    sql_result: Any | None,
    doc_results: list[dict[str, Any]],
    financial_result: Any | None,
    forecast_result: Any | None,
    llm_judge_fn=None,
) -> AnswerFaithfulnessMetrics:
    metrics = AnswerFaithfulnessMetrics()

    import re

    claims = re.split(r'[.!?]+', answer)
    claims = [c.strip() for c in claims if c.strip()]
    metrics.claims_total = len(claims)

    citations = re.findall(r'\[source:\s*(\w+),\s*([^\]]+)\]', answer)
    metrics.citations_total = len(citations)

    sources = {
        "sql": sql_result,
        "document": doc_results,
        "financial": financial_result,
        "forecast": forecast_result,
    }

    supported = 0
    valid_citations = 0

    for claim in claims:
        claim_citations = re.findall(r'\[source:\s*(\w+),\s*([^\]]+)\]', claim)
        if not claim_citations:
            continue

        claim_supported = False
        for source_type, source_id in claim_citations:
            if source_type in sources and sources[source_type]:
                if source_type == "document":
                    for doc in doc_results:
                        if str(doc.get("document_id")) == source_id or doc.get("title") == source_id:
                            valid_citations += 1
                            claim_supported = True
                            break
                else:
                    valid_citations += 1
                    claim_supported = True

        if claim_supported:
            supported += 1

    metrics.claims_supported = supported
    metrics.citations_valid = valid_citations
    metrics.faithfulness_score = supported / metrics.claims_total if metrics.claims_total > 0 else 1.0
    metrics.citation_precision = valid_citations / metrics.citations_total if metrics.citations_total > 0 else 1.0
    metrics.citation_recall = valid_citations / metrics.claims_total if metrics.claims_total > 0 else 1.0
    metrics.hallucination_rate = 1.0 - metrics.faithfulness_score

    return metrics


def evaluate_intent_classification(
    predictions: list[QueryIntent],
    ground_truth: list[QueryIntent],
) -> IntentClassificationMetrics:
    metrics = IntentClassificationMetrics()

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    all_intents = list(QueryIntent)
    intent_names = [i.value for i in all_intents]

    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        if pred == truth:
            correct += 1
        cm_key = truth.value
        if cm_key not in metrics.confusion_matrix:
            metrics.confusion_matrix[cm_key] = {}
        pred_key = pred.value
        metrics.confusion_matrix[cm_key][pred_key] = metrics.confusion_matrix[cm_key].get(pred_key, 0) + 1

    metrics.accuracy = correct / len(predictions) if predictions else 0.0

    for intent_name in intent_names:
        tp = sum(1 for p, t in zip(predictions, ground_truth) if p.value == intent_name and t.value == intent_name)
        fp = sum(1 for p, t in zip(predictions, ground_truth) if p.value == intent_name and t.value != intent_name)
        fn = sum(1 for p, t in zip(predictions, ground_truth) if p.value != intent_name and t.value == intent_name)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.per_class_precision[intent_name] = precision
        metrics.per_class_recall[intent_name] = recall
        metrics.per_class_f1[intent_name] = f1

    return metrics


def evaluate_tool_selection(
    predicted_intents: list[QueryIntent],
    expected_intents: list[QueryIntent],
) -> ToolSelectionMetrics:
    metrics = ToolSelectionMetrics()

    if len(predicted_intents) != len(expected_intents):
        raise ValueError("Predictions and expected must have same length")

    total = len(predicted_intents)
    if total == 0:
        return metrics

    correct = 0
    sql_correct = 0
    sql_total = 0
    rag_correct = 0
    rag_total = 0
    hybrid_correct = 0
    hybrid_total = 0
    financial_correct = 0
    financial_total = 0
    forecast_correct = 0
    forecast_total = 0
    schedule_correct = 0
    schedule_total = 0

    for pred, exp in zip(predicted_intents, expected_intents):
        if pred == exp:
            correct += 1

        if exp in (QueryIntent.SQL, QueryIntent.HYBRID):
            sql_total += 1
            if pred in (QueryIntent.SQL, QueryIntent.HYBRID):
                sql_correct += 1

        if exp in (QueryIntent.DOCUMENT, QueryIntent.HYBRID):
            rag_total += 1
            if pred in (QueryIntent.DOCUMENT, QueryIntent.HYBRID):
                rag_correct += 1

        if exp == QueryIntent.HYBRID:
            hybrid_total += 1
            if pred == QueryIntent.HYBRID:
                hybrid_correct += 1

        if exp == QueryIntent.FINANCIAL_MODEL:
            financial_total += 1
            if pred == QueryIntent.FINANCIAL_MODEL:
                financial_correct += 1

        if exp == QueryIntent.FORECAST:
            forecast_total += 1
            if pred == QueryIntent.FORECAST:
                forecast_correct += 1

        if exp == QueryIntent.SCHEDULE_REPORT:
            schedule_total += 1
            if pred == QueryIntent.SCHEDULE_REPORT:
                schedule_correct += 1

    metrics.accuracy = correct / total
    metrics.sql_vs_rag_correct = (sql_correct / sql_total) if sql_total > 0 else 1.0
    metrics.hybrid_routing_correct = (hybrid_correct / hybrid_total) if hybrid_total > 0 else 1.0
    metrics.financial_model_correct = (financial_correct / financial_total) if financial_total > 0 else 1.0
    metrics.forecast_correct = (forecast_correct / forecast_total) if forecast_total > 0 else 1.0
    metrics.schedule_correct = (schedule_correct / schedule_total) if schedule_total > 0 else 1.0

    return metrics