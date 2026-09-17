import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from nexalens.core.logging import get_logger

logger = get_logger(__name__)

_llm_calls: list[dict[str, Any]] = []
_sql_executions: list[dict[str, Any]] = []
_retrievals: list[dict[str, Any]] = []
_request_latencies: list[float] = []


@dataclass
class LLMMetrics:
    total_calls: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    by_operation: dict[str, dict[str, Any]] = field(default_factory=lambda: defaultdict(lambda: {"count": 0, "tokens": 0, "latency_ms": 0.0}))


@dataclass
class SQLMetrics:
    total_executions: int = 0
    total_latency_ms: float = 0.0
    total_rows: int = 0
    errors: int = 0
    timeouts: int = 0


@dataclass
class RetrievalMetrics:
    total_searches: int = 0
    total_latency_ms: float = 0.0
    total_results: int = 0


def record_llm_call(
    operation: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    status: str = "success",
    error: str | None = None,
) -> None:
    call_data = {
        "operation": operation,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "timestamp": time.time(),
    }
    _llm_calls.append(call_data)

    logger.info(
        "llm_call_recorded",
        operation=operation,
        model=model,
        tokens=call_data["total_tokens"],
        latency_ms=latency_ms,
        status=status,
    )


def record_sql_execution(
    sql: str,
    execution_time_ms: float,
    rows_returned: int,
    status: str = "success",
    error: str | None = None,
) -> None:
    exec_data = {
        "sql": sql[:200],
        "execution_time_ms": execution_time_ms,
        "rows_returned": rows_returned,
        "status": status,
        "error": error,
        "timestamp": time.time(),
    }
    _sql_executions.append(exec_data)

    logger.info(
        "sql_execution_recorded",
        execution_time_ms=execution_time_ms,
        rows_returned=rows_returned,
        status=status,
    )


def record_retrieval(
    query: str,
    results_count: int,
    latency_ms: float,
    reranked: bool = False,
) -> None:
    retrieval_data = {
        "query": query[:100],
        "results_count": results_count,
        "latency_ms": latency_ms,
        "reranked": reranked,
        "timestamp": time.time(),
    }
    _retrievals.append(retrieval_data)

    logger.info(
        "retrieval_recorded",
        results_count=results_count,
        latency_ms=latency_ms,
        reranked=reranked,
    )


def record_request_latency(latency_ms: float) -> None:
    _request_latencies.append(latency_ms)


def get_llm_metrics() -> LLMMetrics:
    metrics = LLMMetrics()
    for call in _llm_calls:
        metrics.total_calls += 1
        metrics.total_tokens += call["total_tokens"]
        metrics.total_latency_ms += call["latency_ms"]
        if call["status"] == "error":
            metrics.errors += 1

        op = call["operation"]
        metrics.by_operation[op]["count"] += 1
        metrics.by_operation[op]["tokens"] += call["total_tokens"]
        metrics.by_operation[op]["latency_ms"] += call["latency_ms"]

    return metrics


def get_sql_metrics() -> SQLMetrics:
    metrics = SQLMetrics()
    for exec in _sql_executions:
        metrics.total_executions += 1
        metrics.total_latency_ms += exec["execution_time_ms"]
        metrics.total_rows += exec["rows_returned"]
        if exec["status"] == "error":
            metrics.errors += 1
        elif exec["status"] == "timeout":
            metrics.timeouts += 1
    return metrics


def get_retrieval_metrics() -> RetrievalMetrics:
    metrics = RetrievalMetrics()
    for ret in _retrievals:
        metrics.total_searches += 1
        metrics.total_latency_ms += ret["latency_ms"]
        metrics.total_results += ret["results_count"]
    return metrics


def get_request_metrics() -> dict[str, float]:
    if not _request_latencies:
        return {"count": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}

    sorted_latencies = sorted(_request_latencies)
    count = len(sorted_latencies)

    def percentile(p: float) -> float:
        idx = int(count * p)
        return sorted_latencies[min(idx, count - 1)]

    return {
        "count": count,
        "avg_ms": sum(sorted_latencies) / count,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def get_metrics_summary() -> dict[str, Any]:
    return {
        "llm": get_llm_metrics().__dict__,
        "sql": get_sql_metrics().__dict__,
        "retrieval": get_retrieval_metrics().__dict__,
        "requests": get_request_metrics(),
    }


def reset_metrics() -> None:
    global _llm_calls, _sql_executions, _retrievals, _request_latencies
    _llm_calls = []
    _sql_executions = []
    _retrievals = []
    _request_latencies = []