"""Unified structured LLM interface for NVIDIA NIM, Gemini, Groq, and OpenAI."""

import json
import logging
import time
import asyncio
import re
from typing import Any, Dict, Optional

import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when no configured provider can complete a structured request."""


class RateLimiter:
    """Sliding-window rate limiter ensuring requests per minute (RPM) do not exceed provider ceilings."""

    def __init__(self, max_rpm: int = 12):
        self.max_rpm = max_rpm
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self):
        if self.max_rpm <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self.max_rpm:
                oldest = self._timestamps[0]
                sleep_needed = 60.0 - (now - oldest) + 0.25
                if sleep_needed > 0:
                    logger.info("Throttling LLM call to respect %d RPM limit: sleeping %.2fs", self.max_rpm, sleep_needed)
                    await asyncio.sleep(sleep_needed)
                    now = time.monotonic()
                    self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            self._timestamps.append(now)


class LLMClient:
    """Dispatch structured JSON requests across configured providers (NVIDIA NIM, Gemini, Groq, OpenAI)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        nvidia_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.nvidia_key = settings.NVIDIA_API_KEY if nvidia_key is None else nvidia_key
        self.nvidia_model = settings.NVIDIA_MODEL
        self.nvidia_base_url = settings.NVIDIA_BASE_URL
        self.openai_key = settings.OPENAI_API_KEY if openai_key is None else openai_key
        self.gemini_key = settings.GEMINI_API_KEY if gemini_key is None else gemini_key
        self.groq_key = settings.GROQ_API_KEY if groq_key is None else groq_key
        self.openai_model = settings.OPENAI_MODEL
        self.gemini_model = settings.GEMINI_MODEL
        self.groq_model = settings.GROQ_MODEL
        self.rate_limiter = RateLimiter(max_rpm=settings.LLM_MAX_RPM)
        self.metrics = {
            "calls": 0,
            "retries": 0,
            "provider_failures": 0,
        }

    @property
    def has_active_key(self) -> bool:
        return bool(self.nvidia_key or self.openai_key or self.gemini_key or self.groq_key)

    def _provider_order(self) -> list[str]:
        configured = {
            "nvidia": bool(self.nvidia_key),
            "gemini": bool(self.gemini_key),
            "groq": bool(self.groq_key),
            "openai": bool(self.openai_key),
        }
        if self.provider != "auto":
            if not configured.get(self.provider, False):
                return []
            return [self.provider] + [
                name for name in ("nvidia", "gemini", "groq", "openai")
                if name != self.provider and configured[name]
            ]
        # In auto mode, prefer NVIDIA (if key provided) > Gemini > Groq > OpenAI
        return [name for name in ("nvidia", "gemini", "groq", "openai") if configured[name]]

    @staticmethod
    def _parse_json_payload(text: str) -> Dict[str, Any]:
        """Robustly parse JSON payload even if wrapped in markdown codeblocks or containing surrounding text."""
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: find outermost curly braces
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_handler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate an object response with bounded retries, multi-model fallback, and provider failover."""
        providers = self._provider_order()
        if not providers:
            raise LLMUnavailableError(
                "No LLM provider is configured. Set NVIDIA_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY."
            )

        errors = []
        for provider in providers:
            for attempt in range(settings.LLM_MAX_RETRIES + 1):
                self.metrics["calls"] += 1
                await self.rate_limiter.acquire()
                started = time.perf_counter()
                try:
                    if provider == "nvidia":
                        result = await self._call_nvidia_json(system_prompt, user_prompt)
                    elif provider == "gemini":
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
                    backoff_delay = settings.LLM_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after and retry_after.strip().isdigit():
                            backoff_delay = float(retry_after.strip()) + 0.5
                        else:
                            backoff_delay = max(6.0, backoff_delay * 8)
                        logger.warning(
                            "Rate limit (429) hit for provider=%s. Backing off for %.2fs before retry %d",
                            provider,
                            backoff_delay,
                            attempt + 1,
                        )
                    self.metrics["retries"] += 1
                    await asyncio.sleep(backoff_delay)

        raise LLMUnavailableError("All configured LLM providers failed: " + "; ".join(errors))

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Retry transport failures and transient provider responses, not bad requests."""
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))

    async def _call_nvidia_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call NVIDIA NIM (OpenAI-compatible) endpoint for DeepSeek or Llama models."""
        payload = {
            "model": self.nvidia_model,
            "messages": [
                {"role": "system", "content": f"{system_prompt}\nCRITICAL: Respond ONLY with a valid, raw JSON object. Do not include markdown preamble, commentary, or conversational explanations."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        headers = {
            "Authorization": f"Bearer {self.nvidia_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self.nvidia_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"NVIDIA NIM ({self.nvidia_model}) returned {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )
            content = response.json()["choices"][0]["message"]["content"]
            return self._parse_json_payload(content)

    async def _call_gemini_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call Google Gemini with automatic multi-model failover on 503 (High Demand) or 404."""
        models_to_try = [self.gemini_model]
        for fallback_m in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        last_error = None
        for model_name in models_to_try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={self.gemini_key}"
            )
            payload = {
                "contents": [{"role": "user", "parts": [{
                    "text": f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"
                }]}],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
            }
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return self._parse_json_payload(text)

                last_error = httpx.HTTPStatusError(
                    f"Gemini ({model_name}) returned {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )
                if response.status_code in (503, 404):
                    logger.warning("Gemini model %s returned %d; trying next fallback model...", model_name, response.status_code)
                    continue
                else:
                    raise last_error

        if last_error:
            raise last_error

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
            return self._parse_json_payload(response.json()["choices"][0]["message"]["content"])

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
            return self._parse_json_payload(response.json()["choices"][0]["message"]["content"])


default_llm_client = LLMClient()
