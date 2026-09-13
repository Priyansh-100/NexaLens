"""
Audit logging for security and compliance.

Provides immutable audit trails for all security-relevant events.
"""

import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import Request
from sqlalchemy import JSON, DateTime, Enum as SQLEnum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from nexalens.core.config import get_settings
from nexalens.core.logging import get_logger
from nexalens.models.session import get_db_session

logger = get_logger(__name__)
settings = get_settings()

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKED = "token_revoked"
    PASSWORD_CHANGED = "password_changed"
    
    # Authorization
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHANGED = "role_changed"
    
    # Data access
    QUERY_EXECUTED = "query_executed"
    DATA_SOURCE_ACCESSED = "data_source_accessed"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DOWNLOADED = "document_downloaded"
    
    # Analytics
    FINANCIAL_MODEL_CREATED = "financial_model_created"
    FORECAST_GENERATED = "forecast_generated"
    REPORT_SCHEDULED = "report_scheduled"
    REPORT_EXECUTED = "report_executed"
    
    # Admin
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    ORGANIZATION_CREATED = "organization_created"
    ORGANIZATION_DELETED = "organization_deleted"
    DATA_SOURCE_CREATED = "data_source_created"
    DATA_SOURCE_DELETED = "data_source_deleted"
    
    # Security
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event."""
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None  # success, failure, denied
    details: Optional[dict] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class AuditLogModel(DeclarativeBase):
    """Database model for audit logs (immutable)."""
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(SQLEnum(AuditEventType), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(SQLEnum(AuditSeverity), nullable=False, index=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    action: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    result: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_audit_logs_user_timestamp", "user_id", "timestamp"),
        Index("ix_audit_logs_org_timestamp", "organization_id", "timestamp"),
        Index("ix_audit_logs_event_timestamp", "event_type", "timestamp"),
    )


class AuditLogger:
    """Service for logging audit events."""

    def __init__(self):
        self._buffer: list[AuditEvent] = []
        self._buffer_size = 100
        self._flush_interval = 5  # seconds
        self._last_flush = time.time()

    async def log(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[dict] = None,
        request: Optional[Request] = None,
    ) -> None:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            organization_id=organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            details=details,
            ip_address=self._get_client_ip(request) if request else None,
            user_agent=request.headers.get("user-agent") if request else None,
            request_id=request_id_var.get() if request_id_var.get() else None,
        )
        
        # Add to buffer
        self._buffer.append(event)
        
        # Flush if buffer full or interval exceeded
        if len(self._buffer) >= self._buffer_size or time.time() - self._last_flush > self._flush_interval:
            await self.flush()

    async def flush(self) -> None:
        """Flush buffered events to database."""
        if not self._buffer:
            return
        
        try:
            async for session in get_db_session():
                for event in self._buffer:
                    log_entry = AuditLogModel(
                        event_type=event.event_type,
                        severity=event.severity,
                        user_id=uuid.UUID(event.user_id) if event.user_id else None,
                        organization_id=uuid.UUID(event.organization_id) if event.organization_id else None,
                        resource_type=event.resource_type,
                        resource_id=event.resource_id,
                        action=event.action,
                        result=event.result,
                        details=event.details,
                        ip_address=event.ip_address,
                        user_agent=event.user_agent,
                        request_id=event.request_id,
                        timestamp=event.timestamp,
                    )
                    session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.error("audit_flush_failed", error=str(e))
        finally:
            self._buffer.clear()
            self._last_flush = time.time()

    def _get_client_ip(self, request: Optional[Request]) -> Optional[str]:
        if not request:
            return None
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else None


# Global audit logger instance
audit_logger = AuditLogger()


async def log_audit_event(
    event_type: AuditEventType,
    severity: AuditSeverity = AuditSeverity.INFO,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[str] = None,
    result: Optional[str] = None,
    details: Optional[dict] = None,
    request: Optional[Request] = None,
) -> None:
    """Convenience function to log an audit event."""
    await audit_logger.log(
        event_type=event_type,
        severity=severity,
        user_id=user_id,
        organization_id=organization_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        result=result,
        details=details,
        request=request,
    )


def get_request_id() -> str:
    """Get current request ID from context."""
    rid = request_id_var.get()
    if not rid:
        rid = str(uuid.uuid4())
        request_id_var.set(rid)
    return rid


def set_request_id(rid: str) -> None:
    """Set request ID in context."""
    request_id_var.set(rid)


class AuditLogQuery:
    """Query builder for audit logs."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._stmt = None

    def filter_by_user(self, user_id: str) -> "AuditLogQuery":
        from sqlalchemy import select
        self._stmt = select(AuditLogModel).where(AuditLogModel.user_id == user_id)
        return self

    def filter_by_organization(self, org_id: str) -> "AuditLogQuery":
        from sqlalchemy import select
        self._stmt = select(AuditLogModel).where(AuditLogModel.organization_id == org_id)
        return self

    def filter_by_event_type(self, event_type: AuditEventType) -> "AuditLogQuery":
        from sqlalchemy import select
        self._stmt = select(AuditLogModel).where(AuditLogModel.event_type == event_type)
        return self

    def filter_by_severity(self, severity: AuditSeverity) -> "AuditLogQuery":
        from sqlalchemy import select
        self._stmt = select(AuditLogModel).where(AuditLogModel.severity == severity)
        return self

    def filter_by_date_range(self, start: datetime, end: datetime) -> "AuditLogQuery":
        from sqlalchemy import select
        if self._stmt is None:
            self._stmt = select(AuditLogModel)
        self._stmt = self._stmt.where(
            AuditLogModel.timestamp >= start,
            AuditLogModel.timestamp <= end,
        )
        return self

    def order_by_timestamp(self, desc: bool = True) -> "AuditLogQuery":
        from sqlalchemy import desc
        if self._stmt is None:
            from sqlalchemy import select
            self._stmt = select(AuditLogModel)
        self._stmt = self._stmt.order_by(desc(AuditLogModel.timestamp) if desc else AuditLogModel.timestamp)
        return self

    def limit(self, limit: int) -> "AuditLogQuery":
        if self._stmt is None:
            from sqlalchemy import select
            self._stmt = select(AuditLogModel)
        self._stmt = self._stmt.limit(limit)
        return self

    async def execute(self) -> list[AuditLogModel]:
        result = await self.session.execute(self._stmt)
        return result.scalars().all()


async def create_audit_log_table() -> None:
    """Create audit log table if not exists."""
    from nexalens.models.session import async_session_maker
    from sqlalchemy import text
    
    async with async_session_maker() as session:
        # Create enum types if not exist
        await session.execute(text("""
            DO $$ BEGIN
                CREATE TYPE audit_event_type AS ENUM (
                    'login_success', 'login_failed', 'logout', 'token_refresh', 
                    'token_revoked', 'password_changed', 'permission_denied', 
                    'role_changed', 'query_executed', 'data_source_accessed',
                    'document_uploaded', 'document_downloaded', 'financial_model_created',
                    'forecast_generated', 'report_scheduled', 'report_executed',
                    'user_created', 'user_deleted', 'organization_created', 
                    'organization_deleted', 'data_source_created', 'data_source_deleted',
                    'rate_limit_exceeded', 'suspicious_activity'
                );
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$;
        """))
        await session.execute(text("""
            DO $$ BEGIN
                CREATE TYPE audit_severity AS ENUM ('info', 'warning', 'error', 'critical');
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$;
        """))
        await session.commit()