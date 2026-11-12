import pytest

from backend.app.orchestration.llm import LLMClient, LLMUnavailableError


@pytest.mark.asyncio
async def test_llm_requires_configured_provider():
    client = LLMClient(provider="groq", groq_key="")
    with pytest.raises(LLMUnavailableError, match="No LLM provider"):
        await client.generate_json("system", "user", fallback_handler=lambda: {"unsafe": True})


@pytest.mark.asyncio
async def test_invalid_configured_provider_does_not_use_heuristic_fallback(monkeypatch):
    client = LLMClient(provider="gemini", gemini_key="invalid")

    async def failure(_system, _user):
        raise ValueError("malformed response")

    monkeypatch.setattr(client, "_call_gemini_json", failure)
    with pytest.raises(LLMUnavailableError, match="All configured"):
        await client.generate_json("system", "user", fallback_handler=lambda: {"unsafe": True})


@pytest.mark.asyncio
async def test_groq_provider_returns_structured_object(monkeypatch):
    client = LLMClient(provider="groq", groq_key="test-key")

    async def fake_call(_system, _user):
        return {"tasks": []}

    monkeypatch.setattr(client, "_call_groq_json", fake_call)
    assert await client.generate_json("system", "user") == {"tasks": []}


@pytest.mark.asyncio
async def test_auto_provider_fails_over_from_gemini_to_groq(monkeypatch):
    client = LLMClient(provider="auto", gemini_key="gemini", groq_key="groq")

    async def gemini_failure(_system, _user):
        raise ValueError("invalid structured response")

    async def groq_success(_system, _user):
        return {"provider": "groq"}

    monkeypatch.setattr(client, "_call_gemini_json", gemini_failure)
    monkeypatch.setattr(client, "_call_groq_json", groq_success)
    assert await client.generate_json("system", "user") == {"provider": "groq"}
