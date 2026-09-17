import contextvars
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from nexalens.core.logging import get_logger

logger = get_logger(__name__)

_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)
_span_stack_var: contextvars.ContextVar[list["Span"]] = contextvars.ContextVar("span_stack", default_factory=list)


@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    parent_id: str | None = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "started"
    error: str | None = None

    def finish(self, status: str = "success", error: str | None = None) -> None:
        self.end_time = time.perf_counter()
        self.status = status
        self.error = error

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


def get_correlation_id() -> str | None:
    return _correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    _correlation_id_var.set(correlation_id)


def clear_correlation_id() -> None:
    _correlation_id_var.set(None)


def get_request_id() -> str | None:
    return _request_id_var.get()


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_user_id() -> str | None:
    return _user_id_var.get()


def set_user_id(user_id: str) -> None:
    _user_id_var.set(user_id)


def get_span_stack() -> list[Span]:
    return _span_stack_var.get()


def push_span(span: Span) -> None:
    stack = get_span_stack()
    stack.append(span)
    _span_stack_var.set(stack)


def pop_span() -> Span | None:
    stack = get_span_stack()
    if stack:
        return stack.pop()
    return None


@asynccontextmanager
async def trace_span(name: str, **metadata: Any) -> AsyncGenerator[Span, None]:
    parent_span = get_span_stack()[-1] if get_span_stack() else None
    span = Span(
        name=name,
        parent_id=parent_span.span_id if parent_span else None,
        metadata=metadata,
    )
    push_span(span)

    correlation_id = get_correlation_id()
    request_id = get_request_id()
    user_id = get_user_id()

    log_data = {
        "span": name,
        "span_id": span.span_id,
        "correlation_id": correlation_id,
        "request_id": request_id,
        "user_id": user_id,
        **metadata,
    }

    logger.debug("span_started", **log_data)

    try:
        yield span
        span.finish("success")
        logger.debug("span_completed", **log_data, duration_ms=span.duration_ms)
    except Exception as e:
        span.finish("error", str(e))
        log_data["error"] = str(e)
        logger.error("span_failed", **log_data, duration_ms=span.duration_ms)
        raise
    finally:
        pop_span()


class CorrelationContext:
    def __init__(self, correlation_id: str | None = None, request_id: str | None = None, user_id: str | None = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.user_id = user_id
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> "CorrelationContext":
        self._tokens.append(_correlation_id_var.set(self.correlation_id))
        self._tokens.append(_request_id_var.set(self.request_id))
        if self.user_id:
            self._tokens.append(_user_id_var.set(self.user_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for token in reversed(self._tokens):
            if token:
                try:
                    _correlation_id_var.reset(token)
                except ValueError:
                    pass


async def create_correlation_context(
    correlation_id: str | None = None,
    request_id: str | None = None,
    user_id: str | None = None,
) -> CorrelationContext:
    ctx = CorrelationContext(correlation_id, request_id, user_id)
    return ctx