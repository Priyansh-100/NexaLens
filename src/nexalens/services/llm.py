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

    def _resolve_temperature(self, temperature: float | None) -> float:
        return self.temperature if temperature is None else temperature

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return self.max_tokens if max_tokens is None else max_tokens

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
                    "temperature": self._resolve_temperature(temperature),
                    "num_predict": self._resolve_max_tokens(max_tokens),
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
                    "temperature": self._resolve_temperature(temperature),
                    "num_predict": self._resolve_max_tokens(max_tokens),
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
        response = await self.generate(full_prompt, system=system, temperature=self._resolve_temperature(temperature), format="json")

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("llm_json_parse_failed", response=response, error=str(e))
            raise LLMError(f"Failed to parse JSON response: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema_class: type,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
    ) -> Any:
        from pydantic import BaseModel

        if not issubclass(schema_class, BaseModel):
            raise ValueError("schema_class must be a Pydantic BaseModel")

        schema_json = schema_class.model_json_schema()
        format_prompt = f"""Return ONLY valid JSON matching this schema:
{json.dumps(schema_json, indent=2)}

No markdown, no explanation, no extra text."""

        full_prompt = f"{prompt}\n\n{format_prompt}"

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.generate(
                    full_prompt,
                    system=system,
                    temperature=self._resolve_temperature(temperature),
                    max_tokens=max_tokens,
                    format="json",
                )

                data = json.loads(response)
                return schema_class(**data)

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("structured_output_json_parse_failed", attempt=attempt + 1, error=str(e))
                if attempt < max_retries - 1:
                    full_prompt = f"{prompt}\n\n{format_prompt}\n\nPrevious attempt failed with JSON parse error: {e}. Please fix and return valid JSON only."
                    continue

            except Exception as e:
                last_error = e
                logger.warning("structured_output_validation_failed", attempt=attempt + 1, error=str(e))
                if attempt < max_retries - 1:
                    full_prompt = f"{prompt}\n\n{format_prompt}\n\nPrevious attempt failed validation: {e}. Please fix and return valid JSON only."
                    continue

        logger.error("structured_output_failed_after_retries", error=str(last_error), max_retries=max_retries)
        raise LLMError(f"Failed to generate valid structured output after {max_retries} attempts: {last_error}") from last_error

    async def health_check(self) -> bool:
        try:
            await self.client.list()
            return True
        except Exception:
            return False


llm_service = LLMService()