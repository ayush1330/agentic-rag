"""
Custom Exception Classes for the FemTech Medical RAG Agent

Provides structured error handling with appropriate HTTP status codes
and consistent error response formatting.
"""

from typing import Any


class AppException(Exception):
    """
    Base exception class for all application-specific exceptions.

    Attributes:
        message: Human-readable error description
        status_code: HTTP status code for the error
        error_code: Machine-readable error identifier
        details: Additional error context
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationError(AppException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class NotFoundError(AppException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource: str = "Resource",
        identifier: str | None = None,
    ) -> None:
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with id '{identifier}' not found"

        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier},
        )


class DatabaseException(AppException):
    """Raised when a database operation fails."""

    def __init__(
        self,
        message: str = "Database operation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details=details,
        )


class ExternalServiceException(AppException):
    """Raised when an external service call fails (Azure, GCP, etc.)."""

    def __init__(
        self,
        service: str,
        message: str = "External service call failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=f"{service}: {message}",
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, **(details or {})},
        )


class AuthenticationError(AppException):
    """Raised when authentication fails (placeholder for Zitadel)."""

    def __init__(
        self,
        message: str = "Authentication required",
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(AppException):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Permission denied",
        required_permission: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details={"required_permission": required_permission} if required_permission else {},
        )


class RateLimitError(AppException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_ERROR",
            details={"retry_after_seconds": retry_after} if retry_after else {},
        )


class MedicalGuardrailError(AppException):
    """
    Raised when a request violates medical safety guardrails.

    This is critical for ensuring the system never provides
    diagnostic or prescriptive medical advice.
    """

    def __init__(
        self,
        violation_type: str,
        message: str = "Request violates medical safety guidelines",
    ) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_code="MEDICAL_GUARDRAIL_VIOLATION",
            details={"violation_type": violation_type},
        )


# =============================================================================
# Cloud Service Exceptions
# =============================================================================


class StorageError(AppException):
    """Raised when GCP Cloud Storage operations fail."""

    def __init__(
        self,
        message: str = "Storage operation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="STORAGE_ERROR",
            details=details,
        )


class DocumentParsingError(AppException):
    """Raised when document parsing fails (LlamaParser or fallback)."""

    def __init__(
        self,
        message: str = "Document parsing failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="DOCUMENT_PARSING_ERROR",
            details=details,
        )


class LLMError(AppException):
    """Raised when Azure OpenAI LLM operations fail."""

    def __init__(
        self,
        message: str = "LLM operation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="LLM_ERROR",
            details=details,
        )


class ObservabilityError(AppException):
    """Raised when observability operations fail."""

    def __init__(
        self,
        message: str = "Observability operation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="OBSERVABILITY_ERROR",
            details=details,
        )
