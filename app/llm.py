import logging
from abc import ABC, abstractmethod
from typing import Optional

import ollama
from google import genai as genai
from google.genai import types

from app.model import Prompt
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: Prompt, model: Optional[str] = None) -> str: ...


class GeminiClient(LLMClient):
    def __init__(self, default_model: str | None = None, api_key: str | None = GEMINI_API_KEY):
        self.default_model = default_model or "gemini-3-flash-preview"
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiClient")
        self._client = genai.Client(api_key=api_key)

    def complete(self, prompt: Prompt, model: Optional[str] = None) -> str:
        model_to_use = model or self.default_model

        try:
            resp = self._client.models.generate_content(
                model=model_to_use,
                config=types.GenerateContentConfig(
                    system_instruction=prompt.system_prompt,
                ),
                contents=prompt.user_prompt,
            )
            return resp.text
        except Exception:
            logger.exception("Gemini completion failed")
            raise


class OllamaClient(LLMClient):
    def __init__(self, default_model: str | None = None):
        self.default_model = default_model or "llama3"

    def complete(self, prompt: Prompt, model: Optional[str] = None) -> str:
        model_to_use = model or self.default_model

        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ]

        try:
            response = ollama.chat(model=model_to_use, messages=messages)
            return response["message"]["content"]
        except Exception:
            logger.exception("Ollama completion failed")
            raise
