from dataclasses import dataclass, field
from typing import Any, Literal

from nexalens.core.config import get_settings
from nexalens.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RAGQualityConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: Literal["fixed", "semantic", "recursive"] = "recursive"

    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    top_k: int = 20
    rerank_top_k: int = 10
    rerank_enabled: bool = False
    rerank_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    metadata_filters: dict[str, Any] = field(default_factory=dict)
    min_score_threshold: float = 0.3

    dedup_threshold: float = 0.9
    dedup_enabled: bool = True

    require_citations: bool = True
    max_citations_per_answer: int = 5
    citation_format: Literal["inline", "footnote", "bracket"] = "bracket"

    confidence_threshold: float = 0.4
    unknown_response: str = "I don't have enough information to answer this question."

    def __post_init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap


DEFAULT_QUALITY_CONFIG = RAGQualityConfig()


class ChunkingEvaluator:
    def __init__(self, config: RAGQualityConfig | None = None):
        self.config = config or DEFAULT_QUALITY_CONFIG

    def evaluate_chunking(
        self,
        documents: list[str],
        ground_truth_chunks: list[list[str]],
    ) -> dict[str, float]:
        from nexalens.documents.processor import document_processor

        results = {}

        for strategy in ["fixed", "semantic", "recursive"]:
            self.config.chunking_strategy = strategy
            # Would need to re-process with different strategies
            # This is a placeholder for the evaluation logic
            results[strategy] = {
                "avg_chunk_size": 0,
                "chunk_count": 0,
                "overlap_ratio": 0,
            }

        return results


class EmbeddingModelConfig:
    AVAILABLE_MODELS = {
        "nomic-embed-text": {"dimension": 768, "max_tokens": 8192},
        "all-MiniLM-L6-v2": {"dimension": 384, "max_tokens": 256},
        "all-mpnet-base-v2": {"dimension": 768, "max_tokens": 384},
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
    }

    def __init__(self, model_name: str = "nomic-embed-text"):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown embedding model: {model_name}")
        self.model_name = model_name
        self.config = self.AVAILABLE_MODELS[model_name]

    @property
    def dimension(self) -> int:
        return self.config["dimension"]

    @property
    def max_tokens(self) -> int:
        return self.config["max_tokens"]


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None and self.model_name:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.warning("sentence_transformers not available, reranking disabled")
                self.model_name = None

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not self.model_name or not documents:
            return documents[:top_k]

        self._load_model()
        if not self._model:
            return documents[:top_k]

        try:
            pairs = [(query, doc.get("content", "")) for doc in documents]
            scores = self._model.predict(pairs)

            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            reranked = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.error("reranking_failed", error=str(e))
            return documents[:top_k]


class MetadataFilter:
    def __init__(self, allowed_filters: dict[str, list[str]] | None = None):
        self.allowed_filters = allowed_filters or {}

    def build_filter(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        if not filters:
            return None

        chroma_filter = {}
        for key, value in filters.items():
            if key not in self.allowed_filters:
                logger.warning("filter_not_allowed", key=key)
                continue

            allowed_values = self.allowed_filters[key]
            if isinstance(value, list):
                valid_values = [v for v in value if v in allowed_values]
                if not valid_values:
                    continue
                if len(valid_values) == 1:
                    chroma_filter[key] = valid_values[0]
                else:
                    chroma_filter[key] = {"$in": valid_values}
            else:
                if value not in allowed_values:
                    continue
                chroma_filter[key] = value

        return chroma_filter if chroma_filter else None


class Deduplicator:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def deduplicate(
        self,
        results: list[dict[str, Any]],
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        if len(results) <= 1:
            return results

        unique = []
        seen_contents = []

        for result in results:
            content = result.get(content_key, "")
            is_duplicate = False

            for seen_content in seen_contents:
                similarity = self._cosine_similarity(content, seen_content)
                if similarity >= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(result)
                seen_contents.append(content)

        logger.debug("deduplication_applied", original=len(results), unique=len(unique))
        return unique

    def _cosine_similarity(self, a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


class CitationTracker:
    def __init__(
        self,
        max_citations: int = 5,
        format: Literal["inline", "footnote", "bracket"] = "bracket",
    ):
        self.max_citations = max_citations
        self.format = format
        self.citations: list[dict[str, Any]] = []

    def add_citation(
        self,
        source_type: str,
        source_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        citation = {
            "type": source_type,
            "id": source_id,
            "title": title,
            "metadata": metadata or {},
        }
        self.citations.append(citation)
        return self._format_citation(citation)

    def _format_citation(self, citation: dict[str, Any]) -> str:
        if self.format == "inline":
            return f"[{citation['type']}: {citation['title'] or citation['id']}]"
        elif self.format == "footnote":
            return f"[{len(self.citations)}]"
        else:
            return f"[source: {citation['type']}, {citation['id']}]"

    def get_citations(self) -> list[dict[str, Any]]:
        return self.citations[:self.max_citations]

    def reset(self) -> None:
        self.citations = []


class ConfidenceEvaluator:
    def __init__(self, threshold: float = 0.4, unknown_response: str | None = None):
        self.threshold = threshold
        self.unknown_response = unknown_response or "I don't have enough information to answer this question."

    def evaluate(
        self,
        sql_result: Any | None,
        doc_results: list[dict[str, Any]],
        financial_result: Any | None,
        forecast_result: Any | None,
    ) -> tuple[float, str | None]:
        scores = []

        if sql_result and getattr(sql_result, 'row_count', 0) > 0:
            scores.append(min(1.0, sql_result.row_count / 10.0) * 0.8 + 0.2)

        if doc_results:
            avg_score = sum(d.get("score", 0) for d in doc_results) / len(doc_results)
            scores.append(avg_score)

        if financial_result:
            scores.append(0.9)

        if forecast_result:
            scores.append(0.85)

        if not scores:
            return 0.3, self.unknown_response

        confidence = sum(scores) / len(scores)

        if confidence < self.threshold:
            return confidence, self.unknown_response

        return confidence, None


rag_quality_config = DEFAULT_QUALITY_CONFIG