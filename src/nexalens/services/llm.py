import json
from collections.abc import AsyncGenerator
from typing import Any

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from nexalens.core.config import get_settings
from nexalens.core.exceptions import LLMError
from nexalens.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.client = ollama.AsyncClient(host=host or settings.ollama_host)
        self.model = model or settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat(
                model=self.model,
                messages=messages,
                format=format,
                options={
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                    **(options or {}),
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e), model=self.model)
            raise LLMError(f"Failed to generate response: {e}") from e

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            async for chunk in await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                },
            ):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        except Exception as e:
            logger.error("llm_stream_failed", error=str(e), model=self.model)
            raise LLMError(f"Failed to stream response: {e}") from e

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        format_prompt = "Return ONLY valid JSON. No markdown, no explanation."
        if schema:
            format_prompt += f" Follow this schema: {json.dumps(schema)}"

        full_prompt = f"{prompt}\n\n{format_prompt}"
        response = await self.generate(full_prompt, system=system, temperature=temperature or 0.0, format="json")

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("llm_json_parse_failed", response=response, error=str(e))
            raise LLMError(f"Failed to parse JSON response: {e}") from e

    async def health_check(self) -> bool:
        try:
            await self.client.list()
            return True
        except Exception:
            return False


llm_service = LLMService()