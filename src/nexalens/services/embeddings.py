from typing import Any

import ollama
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from nexalens.core.config import get_settings
from nexalens.core.exceptions import LLMError
from nexalens.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.client = ollama.AsyncClient(host=host or settings.ollama_host)
        self.model = model or settings.llm_embedding_model
        self._dimension: int | None = None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def embed(self, text: str) -> list[float]:
        try:
            response = await self.client.embeddings(model=self.model, prompt=text)
            embedding = response["embedding"]
            if self._dimension is None:
                self._dimension = len(embedding)
            return embedding
        except Exception as e:
            logger.error("embedding_failed", error=str(e), model=self.model)
            raise LLMError(f"Failed to generate embedding: {e}") from e

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = []
            for text in texts:
                response = await self.client.embeddings(model=self.model, prompt=text)
                embeddings.append(response["embedding"])
            return embeddings
        except Exception as e:
            logger.error("batch_embedding_failed", error=str(e), model=self.model)
            raise LLMError(f"Failed to generate batch embeddings: {e}") from e

    def get_dimension(self) -> int:
        if self._dimension is None:
            raise LLMError("Embedding dimension not known. Call embed() first.")
        return self._dimension

    async def health_check(self) -> bool:
        try:
            await self.client.embeddings(model=self.model, prompt="test")
            return True
        except Exception:
            return False

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vec_a = np.array(a)
        vec_b = np.array(b)
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


embedding_service = EmbeddingService()