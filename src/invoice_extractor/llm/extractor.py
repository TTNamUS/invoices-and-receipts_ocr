"""LLM extraction: OCR text -> structured dict, via LiteLLM.

Wraps the LiteLLM ``completion`` call with retry/backoff and logging. The
3-tier response parser (direct JSON -> regex -> ast.literal_eval) is preserved.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time

from invoice_extractor.config import settings
from invoice_extractor.llm.prompts import SYSTEM_PROMPT, build_messages
from invoice_extractor.logging_config import get_logger

logger = get_logger(__name__)


def _set_api_keys() -> None:
    """Ensure the right API key env var is set for LiteLLM."""
    if settings.llm_provider == "gemini" and settings.google_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
    elif settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)


class LLMExtractor:
    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ):
        self.model = model or settings.llm_model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        _set_api_keys()

    def _complete(self, messages: list) -> str:
        """Call the LLM with retry/backoff. Returns the raw response text."""
        from litellm import completion

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:  # transient API/network errors
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM call failed after %d attempts: %s", self.max_retries, exc)
        raise last_exc  # type: ignore[misc]

    def extract(self, ocr_text: str) -> dict:
        """Send OCR text to the LLM and return the parsed structured dict.

        Raises ValueError if the response cannot be parsed as JSON.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + build_messages(ocr_text)
        raw = self._complete(messages)
        return self._parse_response(raw)

    def _parse_response(self, text: str) -> dict:
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Level 1: direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Level 2: extract JSON object with regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Level 3: Python literal eval (handles single-quote JSON)
        try:
            result = ast.literal_eval(text)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        raise ValueError(f"Cannot parse LLM response as JSON:\n{text[:500]}")

    def extract_with_metadata(self, ocr_text: str, file_name: str = "unknown") -> dict:
        result = self.extract(ocr_text)
        return {
            "file_name": file_name,
            "model": self.model,
            "extraction": result,
        }
