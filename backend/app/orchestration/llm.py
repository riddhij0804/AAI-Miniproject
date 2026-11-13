"""Unified structured LLM interface for Gemini, Groq, and OpenAI."""

import json
import logging
import time
import asyncio
from typing import Any, Dict, Optional

import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when no configured provider can complete a structured request."""


class LLMClient:
    """Dispatch structured JSON requests across configured providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.openai_key = settings.OPENAI_API_KEY if openai_key is None else openai_key
        self.gemini_key = settings.GEMINI_API_KEY if gemini_key is None else gemini_key
        self.groq_key = settings.GROQ_API_KEY if groq_key is None else groq_key
        self.openai_model = settings.OPENAI_MODEL
        self.gemini_model = settings.GEMINI_MODEL
        self.groq_model = settings.GROQ_MODEL
        self.metrics = {
            "calls": 0,
            "retries": 0,
            "provider_failures": 0,
        }

    @property
    def has_active_key(self) -> bool:
        return bool(self.openai_key or self.gemini_key or self.groq_key)

    def _provider_order(self) -> list[str]:
        configured = {
            "gemini": bool(self.gemini_key),
            "groq": bool(self.groq_key),
            "openai": bool(self.openai_key),
        }
        if self.provider != "auto":
            if not configured.get(self.provider, False):
                return []
            return [self.provider] + [
                name for name in ("gemini", "groq", "openai")
                if name != self.provider and configured[name]
            ]
        return [name for name in ("gemini", "groq", "openai") if configured[name]]

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_handler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate an object response with bounded retries and provider fallback."""
        providers = self._provider_order()
        if not providers:
            raise LLMUnavailableError(
                "No LLM provider is configured. Set GEMINI_API_KEY or GROQ_API_KEY."
            )

        errors = []
        for provider in providers:
            for attempt in range(settings.LLM_MAX_RETRIES + 1):
                self.metrics["calls"] += 1
                started = time.perf_counter()
                try:
                    if provider == "gemini":
                        result = await self._call_gemini_json(system_prompt, user_prompt)
                    elif provider == "groq":
                        result = await self._call_groq_json(system_prompt, user_prompt)
                    else:
                        result = await self._call_openai_json(system_prompt, user_prompt)
                    if not isinstance(result, dict):
                        raise ValueError("Provider returned a non-object JSON response")
                    logger.info(
                        "llm_call_complete provider=%s attempt=%d duration_ms=%.2f",
                        provider,
                        attempt + 1,
                        (time.perf_counter() - started) * 1000,
                    )
                    return result
                except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    errors.append(f"{provider} attempt {attempt + 1}: {exc}")
                    self.metrics["provider_failures"] += 1
                    retryable = self._is_retryable(exc)
                    if not retryable or attempt >= settings.LLM_MAX_RETRIES:
                        logger.warning(
                            "llm_call_failed provider=%s attempt=%d retryable=%s error=%s",
                            provider,
                            attempt + 1,
                            retryable,
                            exc,
                        )
                        break
                    self.metrics["retries"] += 1
                    await asyncio.sleep(
                        settings.LLM_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    )
        raise LLMUnavailableError("All configured LLM providers failed: " + "; ".join(errors))

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Retry transport failures and transient provider responses, not bad requests."""
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))

    async def _call_gemini_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{
                "text": f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"
            }]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"Gemini returned {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)

    async def _call_openai_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.openai_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        headers = {"Authorization": "Bearer " + str(self.openai_key), "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"OpenAI returned {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )
            return json.loads(response.json()["choices"][0]["message"]["content"])

    async def _call_groq_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.groq_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        headers = {"Authorization": "Bearer " + str(self.groq_key), "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload
            )
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"Groq returned {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )
            return json.loads(response.json()["choices"][0]["message"]["content"])


default_llm_client = LLMClient()
