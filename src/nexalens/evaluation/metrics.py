from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np


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
class EvaluationResult:
    query_id: UUID
    question: str
    intent: str
    retrieval: RetrievalMetrics | None = None
    sql: SQLMetrics | None = None
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