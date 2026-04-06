import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from phantom.gpt_client import GPTClient, CircuitBreakerOpen


@pytest.fixture
def client():
    return GPTClient(primary_model="gpt-4o", fallback_model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_generate_returns_string_on_success(client):
    with patch.object(client._openai, "chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="hello world"))]
        ))
        result = await client.generate("Say hello", fallback_fn=lambda: "fallback")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_falls_back_to_fallback_fn_on_api_error(client):
    with patch.object(client._openai, "chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        result = await client.generate("Say hello", fallback_fn=lambda: "static fallback")
    assert result == "static fallback"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold(client):
    client._circuit_breaker._threshold = 2
    with patch.object(client._openai, "chat") as mock_chat:
        mock_chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        # First two failures trip the circuit
        await client.generate("q", fallback_fn=lambda: "fb")
        await client.generate("q", fallback_fn=lambda: "fb")
        # Third call should use fallback directly without hitting API
        result = await client.generate("q", fallback_fn=lambda: "circuit open fallback")
    assert result == "circuit open fallback"


def test_circuit_breaker_resets_after_success(client):
    cb = client._circuit_breaker
    cb._failures = 3
    cb._open = True
    cb.record_success()
    assert cb._open is False
    assert cb._failures == 0
