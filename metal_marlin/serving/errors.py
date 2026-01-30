"""Custom exceptions for the Metal Marlin serving layer."""

class ServingError(Exception):
    """Base exception for serving errors."""
    status_code: int = 500
    error_type: str = "server_error"


class ModelNotLoadedError(ServingError):
    """Raised when no model is loaded in the engine."""
    status_code = 503
    error_type = "model_not_loaded"

    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message)


class InvalidRequestError(ServingError):
    """Raised for malformed or invalid requests."""
    status_code = 400
    error_type = "invalid_request"


class ModelNotFoundError(ServingError):
    """Raised when requested model doesn't match loaded model."""
    status_code = 404
    error_type = "model_not_found"


class RateLimitError(ServingError):
    """Raised when rate limit is exceeded."""
    status_code = 429
    error_type = "rate_limit_exceeded"


class ContextLengthExceededError(ServingError):
    """Raised when prompt exceeds max context length."""
    status_code = 400
    error_type = "context_length_exceeded"
