"""core/llm_provider.py -- Real LLM Provider for SatQuery AI.

This module provides a concrete implementation of the LLMProvider protocol
using an OpenAI-compatible API (e.g., Nemotron 3 Ultra Free).

The provider is stateless and handles only:
- Authentication
- HTTP communication
- Request/response formatting
- Timeout handling
- Provider-level error mapping

It does NOT:
- Select SatQuery tools
- Execute tools
- Calculate geospatial indices
- Access raster data
- Modify AnalysisContext
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from core.planner import LLMProvider


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 2

# Environment variable names (no values stored here)
ENV_BASE_URL = "SATQUERY_LLM_BASE_URL"
ENV_API_KEY = "SATQUERY_LLM_API_KEY"
ENV_MODEL = "SATQUERY_LLM_MODEL"
ENV_TIMEOUT = "SATQUERY_LLM_TIMEOUT"


class ProviderConfigurationError(Exception):
    """Raised when provider configuration is invalid or missing."""
    pass


class ProviderError(Exception):
    """Base exception for provider-level errors."""

    def __init__(self, message: str, *, code: str = "PROVIDER_ERROR", status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ProviderTimeoutError(ProviderError):
    """Raised when the provider request times out."""
    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, code="PROVIDER_TIMEOUT", **kwargs)


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication fails."""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, code="PROVIDER_AUTH_ERROR", **kwargs)


class ProviderRateLimitError(ProviderError):
    """Raised when rate limited."""
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, code="PROVIDER_RATE_LIMIT", **kwargs)


class ProviderResponseError(ProviderError):
    """Raised when the provider returns an error response."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="PROVIDER_RESPONSE_ERROR", **kwargs)


class ProviderNetworkError(ProviderError):
    """Raised when a network error occurs."""
    def __init__(self, message: str = "Network error", **kwargs):
        super().__init__(message, code="PROVIDER_NETWORK_ERROR", **kwargs)


class ProviderMalformedResponseError(ProviderError):
    """Raised when the provider response cannot be parsed."""
    def __init__(self, message: str = "Malformed provider response", **kwargs):
        super().__init__(message, code="PROVIDER_MALFORMED_RESPONSE", **kwargs)


# --------------------------------------------------------------------------- #
# NemotronProvider
# --------------------------------------------------------------------------- #

class NemotronProvider:
    """OpenAI-compatible LLM provider for Nemotron 3 Ultra Free (or similar).

    Implements the LLMProvider protocol from core.planner.

    Configuration via environment variables:
    - SATQUERY_LLM_BASE_URL: API base URL (e.g., "https://integrate.api.nvidia.com/v1")
    - SATQUERY_LLM_API_KEY: API key for authentication
    - SATQUERY_LLM_MODEL: Model identifier (e.g., "nvidia/nemotron-3-ultra")
    - SATQUERY_LLM_TIMEOUT: Request timeout in seconds (default: 30)

    The provider uses httpx for HTTP communication with configurable timeouts
    and retries. It maps provider errors to PlannerError codes.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        # Resolve configuration from parameters or environment
        self.base_url = base_url or os.environ.get(ENV_BASE_URL)
        self.api_key = api_key or os.environ.get(ENV_API_KEY)
        self.model = model or os.environ.get(ENV_MODEL)
        self.timeout = timeout or float(os.environ.get(ENV_TIMEOUT, str(DEFAULT_TIMEOUT)))
        self.max_retries = max_retries

        # Validate required configuration
        if not self.base_url:
            raise ProviderConfigurationError(
                f"Missing {ENV_BASE_URL}. Set the API base URL (e.g., 'https://integrate.api.nvidia.com/v1')."
            )
        if not self.api_key:
            raise ProviderConfigurationError(
                f"Missing {ENV_API_KEY}. Set the API key for authentication."
            )
        if not self.model:
            raise ProviderConfigurationError(
                f"Missing {ENV_MODEL}. Set the model identifier (e.g., 'nvidia/nemotron-3-ultra')."
            )

        # Normalize base URL (remove trailing slash, ensure /v1 or similar)
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        # Build the chat completions endpoint
        self.chat_endpoint = f"{self.base_url}/chat/completions"

        # Default headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # httpx client with timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create the httpx client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "NemotronProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the LLM API and return the raw response dict.

        This method implements the LLMProvider protocol. It formats the request
        according to the OpenAI Chat Completions API specification and parses
        the response into the format expected by LLMPlanner.

        Returns:
            Dict matching the OpenAI Chat Completions response format:
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": str | None,
                            "tool_calls": [
                                {
                                    "id": str,
                                    "type": "function",
                                    "function": {"name": str, "arguments": str}
                                }
                            ] | None
                        }
                    }
                ]
            }

        Raises:
            ProviderConfigurationError: If configuration is invalid.
            ProviderTimeoutError: If the request times out.
            ProviderAuthenticationError: If authentication fails.
            ProviderRateLimitError: If rate limited.
            ProviderResponseError: For other HTTP errors.
            ProviderMalformedResponseError: If response cannot be parsed.
        """
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        # Execute request with retries
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._make_request(payload)
                return self._parse_response(response)
            except (ProviderTimeoutError, ProviderAuthenticationError, ProviderRateLimitError, ProviderNetworkError):
                # Don't retry these errors
                raise
            except ProviderResponseError as e:
                # Retry on 5xx errors
                if e.status_code and 500 <= e.status_code < 600:
                    last_exception = e
                    continue
                raise
            except ProviderMalformedResponseError:
                # Don't retry malformed responses
                raise
            except Exception as e:
                last_exception = e
                continue

        # If we exhausted retries
        if last_exception:
            raise ProviderError(
                f"Request failed after {self.max_retries + 1} attempts: {last_exception}",
                code="PROVIDER_MAX_RETRIES_EXCEEDED",
                details={"last_exception": type(last_exception).__name__}
            ) from last_exception

        # Should not reach here
        raise ProviderError("Unexpected provider error", code="PROVIDER_UNKNOWN")

    def _make_request(self, payload: Dict[str, Any]) -> httpx.Response:
        """Make the HTTP request to the provider API."""
        client = self._get_client()
        try:
            response = client.post(
                self.chat_endpoint,
                headers=self.headers,
                json=payload,
            )
            return response
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Request timed out after {self.timeout}s") from e
        except httpx.NetworkError as e:
            raise ProviderNetworkError(f"Network error: {e}") from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected request error: {e}",
                code="PROVIDER_REQUEST_ERROR",
                details={"exception": type(e).__name__}
            ) from e

    def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse the provider response into the expected format."""
        # Handle HTTP error status codes
        if response.status_code != 200:
            self._handle_error_response(response)

        # Parse JSON
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise ProviderMalformedResponseError(
                f"Response is not valid JSON: {e}"
            ) from e

        # Validate response structure
        if not isinstance(data, dict):
            raise ProviderMalformedResponseError("Response is not a JSON object")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ProviderMalformedResponseError("Response missing 'choices' array")

        if len(choices) == 0:
            raise ProviderMalformedResponseError("Response has empty 'choices' array")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderMalformedResponseError("First choice is not an object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ProviderMalformedResponseError("Choice missing 'message' object")

        # Validate message structure
        role = message.get("role")
        if role != "assistant":
            # Not necessarily an error, but log it
            pass

        # Extract content and tool_calls
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        # Ensure tool_calls format if present
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise ProviderMalformedResponseError("'tool_calls' must be an array")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise ProviderMalformedResponseError("Each tool_call must be an object")
                if tc.get("type") != "function":
                    raise ProviderMalformedResponseError("tool_call type must be 'function'")
                func = tc.get("function")
                if not isinstance(func, dict):
                    raise ProviderMalformedResponseError("tool_call.function must be an object")
                if not isinstance(func.get("name"), str):
                    raise ProviderMalformedResponseError("tool_call.function.name must be a string")
                # arguments should be a JSON string
                args = func.get("arguments")
                if args is not None and not isinstance(args, str):
                    raise ProviderMalformedResponseError("tool_call.function.arguments must be a JSON string")

        # Return in the format expected by LLMPlanner
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }]
        }

    def _sanitize_error_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize error details to remove secrets, API keys, tokens, etc."""
        if not details:
            return {}

        sanitized = {}
        secret_patterns = [
            "api_key", "apikey", "api-key", "access_token", "accesstoken", "access-token",
            "authorization", "bearer", "token", "secret", "credential", "password", "key"
        ]

        def sanitize_value(key: str, value: Any) -> Any:
            key_lower = str(key).lower()
            # Check if key suggests it contains a secret
            for pattern in secret_patterns:
                if pattern in key_lower:
                    return "[REDACTED]"

            # Recursively sanitize dicts and lists
            if isinstance(value, dict):
                return {k: sanitize_value(k, v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(str(i), v) for i, v in enumerate(value)]
            elif isinstance(value, str):
                # Check if the string value looks like a secret (e.g., long alphanumeric)
                if len(value) > 20 and value.isalnum():
                    return "[REDACTED]"
                return value
            return value

        for k, v in details.items():
            sanitized[k] = sanitize_value(k, v)

        return sanitized

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Convert HTTP error responses to appropriate ProviderError."""
        status = response.status_code
        error_details = {}

        # Try to extract error details from response
        try:
            error_data = response.json()
            error_details = error_data if isinstance(error_data, dict) else {}
        except Exception:
            error_details = {"raw_body": response.text[:500]}

        # Sanitize error details to remove secrets
        error_details = self._sanitize_error_details(error_details)

        if status == 401:
            raise ProviderAuthenticationError(
                "Invalid API key or authentication failed",
                status_code=status,
                details=error_details,
            )
        elif status == 429:
            raise ProviderRateLimitError(
                "Rate limit exceeded",
                status_code=status,
                details=error_details,
            )
        elif status == 400:
            raise ProviderResponseError(
                f"Bad request: {error_details.get('error', {}).get('message', 'Invalid request')}",
                status_code=status,
                details=error_details,
            )
        elif 500 <= status < 600:
            raise ProviderResponseError(
                f"Provider server error ({status})",
                status_code=status,
                details=error_details,
            )
        else:
            raise ProviderResponseError(
                f"HTTP {status}: {response.text[:200]}",
                status_code=status,
                details=error_details,
            )


# --------------------------------------------------------------------------- #
# Provider Factory
# --------------------------------------------------------------------------- #

def create_provider_from_env() -> Optional[NemotronProvider]:
    """Create a NemotronProvider from environment variables.

    Returns None if configuration is not available (allowing fallback to MockLLMProvider).
    """
    base_url = os.environ.get(ENV_BASE_URL)
    api_key = os.environ.get(ENV_API_KEY)
    model = os.environ.get(ENV_MODEL)

    if not base_url or not api_key or not model:
        return None

    try:
        return NemotronProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
    except ProviderConfigurationError:
        return None


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "NemotronProvider",
    "create_provider_from_env",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderNetworkError",
    "ProviderResponseError",
    "ProviderMalformedResponseError",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "ENV_MODEL",
    "ENV_TIMEOUT",
]
