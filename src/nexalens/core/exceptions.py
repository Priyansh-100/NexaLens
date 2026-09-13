from typing import Any


class NexaLensError(Exception):
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ConfigurationError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class DatabaseError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="DATABASE_ERROR", details=details)


class VectorStoreError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="VECTOR_STORE_ERROR", details=details)


class LLMError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="LLM_ERROR", details=details)


class SQLGenerationError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="SQL_GENERATION_ERROR", details=details)


class SQLExecutionError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="SQL_EXECUTION_ERROR", details=details)


class DocumentProcessingError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="DOCUMENT_PROCESSING_ERROR", details=details)


class RetrievalError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="RETRIEVAL_ERROR", details=details)


class AuthenticationError(NexaLensError):
    def __init__(self, message: str = "Authentication failed", details: dict[str, Any] | None = None):
        super().__init__(message, code="AUTHENTICATION_ERROR", details=details)


class AuthorizationError(NexaLensError):
    def __init__(self, message: str = "Insufficient permissions", details: dict[str, Any] | None = None):
        super().__init__(message, code="AUTHORIZATION_ERROR", details=details)


class ValidationError(NexaLensError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class NotFoundError(NexaLensError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            f"{resource} not found: {identifier}",
            code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier},
        )