from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DataSourceType(str, Enum):
    SQL = "sql"
    DOCUMENT = "document"


class QueryIntent(str, Enum):
    SQL = "sql"
    DOCUMENT = "document"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Organization(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    slug: str
    description: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class OrganizationCreate(BaseModel):
    name: str
    slug: str
    description: str | None = None


class OrganizationUpdate(BaseModel):
    name: str | None = None
    slug: str | None = None
    description: str | None = None
    is_active: bool | None = None


class FinancialModelType(str, Enum):
    DCF = "dcf"
    NPV = "npv"
    IRR = "irr"
    PAYBACK = "payback"
    SENSITIVITY = "sensitivity"


class ForecastModelType(str, Enum):
    PROPHET = "prophet"
    ARIMA = "arima"
    ETS = "ets"


class ReportFormat(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    HTML = "html"


class QueryIntent(str, Enum):
    SQL = "sql"
    DOCUMENT = "document"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"
    FINANCIAL_MODEL = "financial_model"
    FORECAST = "forecast"
    SCHEDULE_REPORT = "schedule_report"


class DataSource(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    type: DataSourceType
    config: dict[str, Any]
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class QueryRequest(BaseModel):
    question: str
    data_source_ids: list[UUID] | None = None
    intent: QueryIntent | None = None
    max_results: int = 10
    include_sql: bool = True
    include_sources: bool = True


class SQLResult(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float
    explanation: str | None = None


class DocumentResult(BaseModel):
    document_id: UUID
    title: str
    chunk_content: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    question: str
    intent: QueryIntent
    answer: str
    sql_result: SQLResult | None = None
    document_results: list[DocumentResult] = Field(default_factory=list)
    financial_result: FinancialModelResult | None = None
    forecast_result: ForecastResult | None = None
    schedule_result: ReportSchedule | None = None
    confidence: float = 0.0
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class User(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    email: str
    name: str
    role: UserRole = UserRole.VIEWER
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str
    role: UserRole
    exp: int
    type: Literal["access", "refresh"] = "access"


class FinancialModelRequest(BaseModel):
    model_type: FinancialModelType
    assumptions: dict[str, Any]
    cash_flows: list[float] | None = None
    time_periods: int = 5
    output_format: Literal["json", "excel"] = "excel"


class FinancialModelResult(BaseModel):
    model_type: FinancialModelType
    npv: float | None = None
    irr: float | None = None
    payback_period: float | None = None
    dcf_value: float | None = None
    sensitivity_table: dict[str, list[float]] | None = None
    assumptions_used: dict[str, Any]
    excel_base64: str | None = None
    explanation: str


class ForecastRequest(BaseModel):
    table_name: str
    metric_column: str
    date_column: str
    periods: int = 12
    frequency: Literal["D", "W", "M", "Q", "Y"] = "M"
    confidence_interval: float = 0.95
    include_holidays: bool = True
    model_type: ForecastModelType = ForecastModelType.PROPHET


class ForecastPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResult(BaseModel):
    forecast: list[ForecastPoint]
    metrics: dict[str, float]
    model_type: str
    parameters: dict[str, Any]
    plot_base64: str | None = None
    explanation: str


class ReportSchedule(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    query: str
    data_source_ids: list[UUID]
    cron_expression: str
    recipients: list[str]
    format: ReportFormat = ReportFormat.PDF
    template: str | None = None
    is_active: bool = True
    created_by: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_run: datetime | None = None
    next_run: datetime | None = None


class ReportScheduleCreate(BaseModel):
    name: str
    query: str
    data_source_ids: list[UUID]
    cron_expression: str
    recipients: list[str]
    format: ReportFormat = ReportFormat.PDF
    template: str | None = None


class ReportScheduleUpdate(BaseModel):
    name: str | None = None
    query: str | None = None
    data_source_ids: list[UUID] | None = None
    cron_expression: str | None = None
    recipients: list[str] | None = None
    format: ReportFormat | None = None
    template: str | None = None
    is_active: bool | None = None


class ReportExecution(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    schedule_id: UUID
    status: Literal["success", "failed", "running"]
    output_path: str | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: dict[str, bool]