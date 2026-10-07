"""Unified LLM interface supporting Gemini, OpenAI, and deterministic heuristic fallback."""

import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Dispatches calls to Gemini, OpenAI, or falls back to domain-independent heuristic models."""

    def __init__(
        self,
        provider: Optional[str] = None,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.openai_key = openai_key or settings.OPENAI_API_KEY
        self.gemini_key = gemini_key or settings.GEMINI_API_KEY
        self.openai_model = settings.OPENAI_MODEL
        self.gemini_model = settings.GEMINI_MODEL

    @property
    def has_active_key(self) -> bool:
        return bool(self.openai_key or self.gemini_key)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_handler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON output from LLM, or invoke fallback_handler if unavailable/fails."""
        # 1. Try Gemini if configured
        if self.gemini_key and (self.provider in ("auto", "gemini")):
            try:
                res = await self._call_gemini_json(system_prompt, user_prompt)
                if res is not None:
                    return res
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Attempting fallback.")

        # 2. Try OpenAI if configured
        if self.openai_key and (self.provider in ("auto", "openai")):
            try:
                res = await self._call_openai_json(system_prompt, user_prompt)
                if res is not None:
                    return res
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}. Attempting fallback.")

        # 3. Deterministic fallback
        if fallback_handler:
            return fallback_handler()
        return {}

    async def _call_gemini_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call Gemini generateContent REST endpoint requesting JSON response."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"SYSTEM INSTRUCTION:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            else:
                logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                return None

    async def _call_openai_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI chat completions REST endpoint requesting JSON response."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                logger.warning(f"OpenAI API returned status {resp.status_code}: {resp.text}")
                return None


default_llm_client = LLMClient()

