from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from nexalens.models.schemas import (
    FinancialModelType,
    ForecastModelType,
    QueryIntent,
    ReportFormat,
)


class IntentClassification(BaseModel):
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SQLGeneration(BaseModel):
    sql: str
    explanation: str
    tables_used: list[str]
    estimated_complexity: Literal["low", "medium", "high"] = "medium"


class FinancialModelExtraction(BaseModel):
    model_type: FinancialModelType
    assumptions: dict[str, float] = Field(default_factory=dict)
    cash_flows: list[float] | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class TimeSeriesIdentification(BaseModel):
    table: str
    metric_column: str
    date_column: str
    confidence: float = Field(ge=0.0, le=1.0)
    frequency_hint: str | None = None


class ScheduleExtraction(BaseModel):
    name: str
    query: str
    data_source_ids: list[UUID]
    cron_expression: str
    recipients: list[str]
    format: ReportFormat = ReportFormat.PDF
    template: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class DocumentSearchResult(BaseModel):
    document_id: UUID
    title: str
    chunk_content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerSynthesis(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    key_insights: list[str] = Field(default_factory=list)
    data_sufficiency: Literal["sufficient", "partial", "insufficient"] = "sufficient"


class LLMUsage(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    operation: str
    prompt_version: str
    temperature: float
    max_tokens: int


class SQLExecutionMetrics(BaseModel):
    sql: str
    execution_time_ms: float
    rows_returned: int
    columns: list[str]
    cost_estimate: float | None = None
    status: Literal["success", "timeout", "error", "blocked"] = "success"
    error: str | None = None


class RetrievalMetrics(BaseModel):
    query_embedding_dim: int
    retrieved_count: int
    reranked: bool
    top_scores: list[float]
    latency_ms: float
    filter_applied: dict[str, Any] | None = None


class CorrelationContext(BaseModel):
    correlation_id: str
    request_id: str
    user_id: str | None = None
    start_time: datetime = Field(default_factory=datetime.utcnow)
    spans: list[dict[str, Any]] = Field(default_factory=list)


class IntentRouterConfig(BaseModel):
    prompt_version: str = "1.0"
    temperature: float = 0.0
    max_tokens: int = 50
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)


class SQLAgentConfig(BaseModel):
    prompt_version: str = "1.0"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_rows: int = 1000
    timeout_seconds: int = 30
    max_query_cost: float = 10000.0
    max_query_length: int = 50000


class RAGRetrieverConfig(BaseModel):
    prompt_version: str = "1.0"
    top_k: int = 20
    rerank_top_k: int = 10
    min_score_threshold: float = 0.3
    rerank_enabled: bool = False
    rerank_model: str | None = None
    dedup_threshold: float = 0.9
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class AnswerSynthesizerConfig(BaseModel):
    prompt_version: str = "1.0"
    temperature: float = 0.2
    max_tokens: int = 1024
    require_citations: bool = True
    max_citations_per_answer: int = 5
    confidence_threshold: float = 0.4
    unknown_response: str = "I don't have enough information to answer this question."


from typing import Literal