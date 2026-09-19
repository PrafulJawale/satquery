"""tests/test_llm_provider.py -- Tests for the Nemotron LLM Provider.

Tests use mocked HTTP responses to avoid depending on the real Nemotron API.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import httpx
import pytest

from core.llm_provider import (
    NemotronProvider,
    create_provider_from_env,
    ProviderConfigurationError,
    ProviderTimeoutError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderMalformedResponseError,
    ProviderError,
    ENV_BASE_URL,
    ENV_API_KEY,
    ENV_MODEL,
)
from core.planner import LLMProvider


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def clear_env():
    """Clear LLM-related environment variables before each test."""
    for key in [ENV_BASE_URL, ENV_API_KEY, ENV_MODEL, "SATQUERY_LLM_TIMEOUT"]:
        if key in os.environ:
            del os.environ[key]
    yield
    for key in [ENV_BASE_URL, ENV_API_KEY, ENV_MODEL, "SATQUERY_LLM_TIMEOUT"]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for testing."""
    with patch("core.llm_provider.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def valid_config():
    """Valid provider configuration."""
    return {
        "base_url": "https://api.example.com/v1",
        "api_key": "test-api-key",
        "model": "test-model",
    }


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the NDVI?"},
    ]


@pytest.fixture
def sample_tools():
    """Sample tools schema for testing."""
    return [{
        "type": "function",
        "function": {
            "name": "compute_ndvi",
            "description": "Compute NDVI for an area",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}
        }
    }]


# --------------------------------------------------------------------------- #
# 1. Configuration Tests
# --------------------------------------------------------------------------- #

def test_create_provider_missing_base_url(valid_config):
    """Provider creation fails without base_url."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        NemotronProvider(api_key=valid_config["api_key"], model=valid_config["model"])
    assert "SATQUERY_LLM_BASE_URL" in str(exc_info.value)


def test_create_provider_missing_api_key(valid_config):
    """Provider creation fails without api_key."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        NemotronProvider(base_url=valid_config["base_url"], model=valid_config["model"])
    assert "SATQUERY_LLM_API_KEY" in str(exc_info.value)


def test_create_provider_missing_model(valid_config):
    """Provider creation fails without model."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        NemotronProvider(base_url=valid_config["base_url"], api_key=valid_config["api_key"])
    assert "SATQUERY_LLM_MODEL" in str(exc_info.value)


def test_create_provider_from_env_missing():
    """create_provider_from_env returns None when config is missing."""
    result = create_provider_from_env()
    assert result is None


def test_create_provider_from_env_valid(monkeypatch, valid_config):
    """create_provider_from_env returns provider when env vars are set."""
    monkeypatch.setenv(ENV_BASE_URL, valid_config["base_url"])
    monkeypatch.setenv(ENV_API_KEY, valid_config["api_key"])
    monkeypatch.setenv(ENV_MODEL, valid_config["model"])

    provider = create_provider_from_env()
    assert provider is not None
    assert isinstance(provider, NemotronProvider)
    # create_provider_from_env normalizes the URL - if it already ends with /v1, it's preserved
    expected_base = "https://api.example.com/v1"
    assert provider.base_url == expected_base
    assert provider.api_key == valid_config["api_key"]
    assert provider.model == valid_config["model"]


def test_create_provider_from_env_partial(monkeypatch, valid_config):
    """create_provider_from_env returns None when only some env vars are set."""
    monkeypatch.setenv(ENV_BASE_URL, valid_config["base_url"])
    # Missing API key and model

    result = create_provider_from_env()
    assert result is None


def test_base_url_normalization():
    """Base URL is normalized to end with /v1."""
    provider = NemotronProvider(
        base_url="https://api.example.com",
        api_key="key",
        model="model",
    )
    assert provider.base_url == "https://api.example.com/v1"
    assert provider.chat_endpoint == "https://api.example.com/v1/chat/completions"

    provider2 = NemotronProvider(
        base_url="https://api.example.com/v1/",
        api_key="key",
        model="model",
    )
    assert provider2.base_url == "https://api.example.com/v1"


def test_timeout_from_env(monkeypatch, valid_config):
    """Timeout is read from environment variable."""
    monkeypatch.setenv(ENV_BASE_URL, valid_config["base_url"])
    monkeypatch.setenv(ENV_API_KEY, valid_config["api_key"])
    monkeypatch.setenv(ENV_MODEL, valid_config["model"])
    monkeypatch.setenv("SATQUERY_LLM_TIMEOUT", "45")

    provider = create_provider_from_env()
    assert provider.timeout == 45.0


# --------------------------------------------------------------------------- #
# 2. Successful Request Tests
# --------------------------------------------------------------------------- #

def test_complete_success_tool_call(mock_httpx_client, valid_config, sample_messages, sample_tools):
    """Successful response with tool_call is parsed correctly."""
    provider = NemotronProvider(**valid_config)

    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "compute_ndvi",
                        "arguments": '{"query": "What is the NDVI?"}'
                    }
                }]
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    result = provider.complete(sample_messages, tools=sample_tools)

    assert "choices" in result
    assert len(result["choices"]) == 1
    message = result["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["tool_calls"] is not None
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "compute_ndvi"
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["query"] == "What is the NDVI?"


def test_complete_success_clarification(mock_httpx_client, valid_config, sample_messages):
    """Successful response with plain text clarification is parsed correctly."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Please select an area on the map first.",
                "tool_calls": None
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    result = provider.complete(sample_messages)

    assert "choices" in result
    message = result["choices"][0]["message"]
    assert message["content"] == "Please select an area on the map first."
    assert message["tool_calls"] is None


def test_complete_passes_parameters(mock_httpx_client, valid_config, sample_messages, sample_tools):
    """Request parameters are passed correctly to the API."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "OK",
                "tool_calls": None
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    provider.complete(
        sample_messages,
        temperature=0.5,
        max_tokens=1000,
        tools=sample_tools,
        tool_choice="auto",
    )

    # Verify the request payload
    call_args = mock_httpx_client.post.call_args
    assert call_args is not None
    payload = call_args[1]["json"]  # httpx uses json= for JSON body

    assert payload["model"] == valid_config["model"]
    assert payload["messages"] == sample_messages
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 1000
    assert payload["tools"] == sample_tools
    assert payload["tool_choice"] == "auto"


def test_complete_headers(mock_httpx_client, valid_config, sample_messages):
    """Request includes correct headers."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "OK", "tool_calls": None}}]
    }
    mock_httpx_client.post.return_value = mock_response

    provider.complete(sample_messages)

    call_args = mock_httpx_client.post.call_args
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == f"Bearer {valid_config['api_key']}"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"


# --------------------------------------------------------------------------- #
# 3. Error Handling Tests
# --------------------------------------------------------------------------- #

def test_timeout_error(mock_httpx_client, valid_config, sample_messages):
    """Timeout is converted to ProviderTimeoutError."""
    provider = NemotronProvider(**valid_config)

    mock_httpx_client.post.side_effect = httpx.TimeoutException("Request timed out")

    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_TIMEOUT"
    assert "timed out" in str(exc_info.value).lower()


def test_network_error(mock_httpx_client, valid_config, sample_messages):
    """Network errors are converted to ProviderError."""
    provider = NemotronProvider(**valid_config)

    mock_httpx_client.post.side_effect = httpx.NetworkError("DNS resolution failed")

    with pytest.raises(ProviderError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_NETWORK_ERROR"


def test_authentication_error_401(mock_httpx_client, valid_config, sample_messages):
    """401 response raises ProviderAuthenticationError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_AUTH_ERROR"
    assert exc_info.value.status_code == 401


def test_rate_limit_error_429(mock_httpx_client, valid_config, sample_messages):
    """429 response raises ProviderRateLimitError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_RATE_LIMIT"
    assert exc_info.value.status_code == 429


def test_bad_request_400(mock_httpx_client, valid_config, sample_messages):
    """400 response raises ProviderResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": {"message": "Invalid model"}}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_RESPONSE_ERROR"
    assert exc_info.value.status_code == 400


def test_server_error_500_retries(mock_httpx_client, valid_config, sample_messages):
    """500 errors are retried (up to max_retries)."""
    provider = NemotronProvider(**valid_config, max_retries=2)

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": {"message": "Internal server error"}}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderError) as exc_info:
        provider.complete(sample_messages)

    # Should have retried max_retries + 1 times
    assert mock_httpx_client.post.call_count == 3
    assert exc_info.value.code == "PROVIDER_MAX_RETRIES_EXCEEDED"


def test_server_error_503_no_retry_on_success(mock_httpx_client, valid_config, sample_messages):
    """503 followed by success works."""
    provider = NemotronProvider(**valid_config, max_retries=2)

    # First call fails with 503
    error_response = Mock()
    error_response.status_code = 503
    error_response.json.return_value = {"error": {"message": "Service unavailable"}}

    # Second call succeeds
    success_response = Mock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "OK", "tool_calls": None}}]
    }

    mock_httpx_client.post.side_effect = [error_response, success_response]

    result = provider.complete(sample_messages)
    assert result["choices"][0]["message"]["content"] == "OK"
    assert mock_httpx_client.post.call_count == 2


def test_malformed_json_response(mock_httpx_client, valid_config, sample_messages):
    """Invalid JSON response raises ProviderMalformedResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderMalformedResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_MALFORMED_RESPONSE"


def test_empty_choices(mock_httpx_client, valid_config, sample_messages):
    """Empty choices array raises ProviderMalformedResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderMalformedResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_MALFORMED_RESPONSE"


def test_missing_message(mock_httpx_client, valid_config, sample_messages):
    """Missing message in choice raises ProviderMalformedResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{}]}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderMalformedResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_MALFORMED_RESPONSE"


def test_invalid_tool_calls_format(mock_httpx_client, valid_config, sample_messages):
    """Invalid tool_calls format raises ProviderMalformedResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": "not-an-array"
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderMalformedResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_MALFORMED_RESPONSE"


def test_invalid_function_name(mock_httpx_client, valid_config, sample_messages):
    """Non-string function name raises ProviderMalformedResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": 123, "arguments": "{}"}
                }]
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderMalformedResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_MALFORMED_RESPONSE"


def test_arguments_not_string(mock_httpx_client, valid_config, sample_messages):
    """Non-string arguments raises ProviderMalformedResponseError."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "compute_ndvi", "arguments": {"query": "test"}}
                }]
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderMalformedResponseError) as exc_info:
        provider.complete(sample_messages)

    assert exc_info.value.code == "PROVIDER_MALFORMED_RESPONSE"


# --------------------------------------------------------------------------- #
# 4. Provider Protocol Compliance
# --------------------------------------------------------------------------- #

def test_provider_implements_protocol(valid_config):
    """NemotronProvider implements the LLMProvider protocol."""
    provider = NemotronProvider(**valid_config)
    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "complete")
    assert callable(provider.complete)


def test_complete_signature_matches_protocol(valid_config):
    """complete() method signature matches LLMProvider protocol."""
    import inspect

    provider = NemotronProvider(**valid_config)
    sig = inspect.signature(provider.complete)

    params = sig.parameters
    assert "messages" in params
    assert "temperature" in params
    assert "max_tokens" in params
    assert "tools" in params
    assert "tool_choice" in params

    # Check defaults
    assert params["temperature"].default == 0.0
    assert params["max_tokens"].default == 512
    assert params["tools"].default is None
    assert params["tool_choice"].default is None


# --------------------------------------------------------------------------- #
# 5. Integration with Planner
# --------------------------------------------------------------------------- #

def test_provider_integration_with_planner(valid_config, sample_messages):
    """Provider can be used with LLMPlanner."""
    from core.planner import LLMPlanner
    from analyses.base import AnalysisContext

    provider = NemotronProvider(**valid_config)
    ctx = AnalysisContext(
        roi=None, ndvi=None, ndvi_confirmed=False, raster_label="test.tif"
    )
    planner = LLMPlanner(provider, ctx)

    # Verify planner can call provider.complete
    with patch.object(provider, "complete", return_value={
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Please select an area on the map first.",
                "tool_calls": None
            }
        }]
    }) as mock_complete:
        response = planner.plan("What is NDVI?")
        mock_complete.assert_called_once()
        assert response.has_clarification


# --------------------------------------------------------------------------- #
# 6. Credential Safety
# --------------------------------------------------------------------------- #

def test_api_key_not_in_error_messages(mock_httpx_client, valid_config, sample_messages):
    """API key is not exposed in error messages."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        provider.complete(sample_messages)

    error_str = str(exc_info.value)
    assert "test-api-key" not in error_str
    assert "Bearer" not in error_str


def test_api_key_not_in_details(mock_httpx_client, valid_config, sample_messages):
    """API key is not in error details."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
    mock_httpx_client.post.return_value = mock_response

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        provider.complete(sample_messages)

    details = exc_info.value.details
    # Details should not contain the API key
    details_str = str(details)
    assert "test-api-key" not in details_str


def test_credentials_not_in_logs(mock_httpx_client, valid_config, sample_messages, capfd):
    """Credentials don't appear in any printed output."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "OK", "tool_calls": None}}]
    }
    mock_httpx_client.post.return_value = mock_response

    provider.complete(sample_messages)

    out, err = capfd.readouterr()
    assert "test-api-key" not in out
    assert "test-api-key" not in err


# --------------------------------------------------------------------------- #
# 7. Context Manager
# --------------------------------------------------------------------------- #

def test_context_manager_closes_client(valid_config):
    """Context manager properly closes the HTTP client."""
    with patch("core.llm_provider.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with NemotronProvider(**valid_config) as provider:
            _ = provider._get_client()

        mock_client.close.assert_called_once()


def test_explicit_close(valid_config):
    """Explicit close() works."""
    with patch("core.llm_provider.httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = NemotronProvider(**valid_config)
        _ = provider._get_client()
        provider.close()

        mock_client.close.assert_called_once()


# --------------------------------------------------------------------------- #
# 8. Retry Behavior
# --------------------------------------------------------------------------- #

def test_max_retries_configurable(valid_config):
    """max_retries is configurable."""
    provider = NemotronProvider(**valid_config, max_retries=5)
    assert provider.max_retries == 5

    provider2 = NemotronProvider(**valid_config)  # default
    assert provider2.max_retries == 2


def test_timeout_configurable(valid_config):
    """timeout is configurable."""
    provider = NemotronProvider(**valid_config, timeout=60.0)
    assert provider.timeout == 60.0

    provider2 = NemotronProvider(**valid_config)  # default
    assert provider2.timeout == 30.0


# --------------------------------------------------------------------------- #
# 9. Empty Response Handling
# --------------------------------------------------------------------------- #

def test_empty_content_with_no_tool_calls(mock_httpx_client, valid_config, sample_messages):
    """Empty content with no tool_calls is handled."""
    provider = NemotronProvider(**valid_config)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": None
            }
        }]
    }
    mock_httpx_client.post.return_value = mock_response

    result = provider.complete(sample_messages)
    # Empty string content is valid - planner will treat as no tool call and no clarification
    assert result["choices"][0]["message"]["content"] == ""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
