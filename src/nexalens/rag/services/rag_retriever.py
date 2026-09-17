import time
from typing import Any
from uuid import UUID

from nexalens.core.config import get_settings
from nexalens.core.exceptions import RetrievalError
from nexalens.core.logging import get_logger
from nexalens.documents.service import document_service
from nexalens.rag.quality import (
    CitationTracker,
    ConfidenceEvaluator,
    Deduplicator,
    MetadataFilter,
    RAGQualityConfig,
    Reranker,
)
from nexalens.rag.schemas import (
    DocumentSearchResult,
    LLMUsage,
    RAGRetrieverConfig,
    RetrievalMetrics,
)
from nexalens.services.embeddings import embedding_service

logger = get_logger(__name__)
settings = get_settings()


class RAGRetriever:
    def __init__(self, config: RAGRetrieverConfig | None = None):
        self.config = config or RAGRetrieverConfig()
        self.embedding_service = embedding_service
        self.document_service = document_service

        self.quality_config = RAGQualityConfig()
        self.deduplicator = Deduplicator(threshold=self.quality_config.dedup_threshold)
        self.reranker = Reranker(model_name=self.quality_config.rerank_model)
        self.metadata_filter = MetadataFilter()
        self.citation_tracker = CitationTracker(
            max_citations=self.quality_config.max_citations_per_answer,
            format=self.quality_config.citation_format,
        )
        self.confidence_evaluator = ConfidenceEvaluator(
            threshold=self.quality_config.confidence_threshold,
            unknown_response=self.quality_config.unknown_response,
        )

    async def search_documents(
        self,
        question: str,
        top_k: int | None = None,
        source_ids: list[UUID] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[DocumentSearchResult]:
        start_time = time.perf_counter()
        k = top_k or self.config.top_k

        try:
            query_embedding = await self.embedding_service.embed(question)

            filter_dict = None
            if source_ids:
                filter_dict = {"source_id": {"$in": [str(sid) for sid in source_ids]}}

            if metadata_filters:
                chroma_filter = self.metadata_filter.build_filter(metadata_filters)
                if chroma_filter:
                    if filter_dict:
                        filter_dict = {"$and": [filter_dict, chroma_filter]}
                    else:
                        filter_dict = chroma_filter

            chunks = await self.document_service.search_documents(
                query_embedding,
                top_k=k,
                source_ids=source_ids,
            )

            results = [
                DocumentSearchResult(
                    document_id=chunk["metadata"]["document_id"],
                    title=chunk["metadata"].get("document_title", "Unknown"),
                    chunk_content=chunk["content"][:500],
                    score=chunk["score"],
                    metadata=chunk["metadata"],
                )
                for chunk in chunks
            ]

            if self.quality_config.min_score_threshold > 0:
                results = [r for r in results if r.score >= self.quality_config.min_score_threshold]

            if self.quality_config.dedup_enabled:
                results = self._deduplicate(results)

            if self.quality_config.rerank_enabled and results:
                results = await self._rerank(question, results)

            results = results[: self.quality_config.rerank_top_k]

            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "document_search_completed",
                query=question[:100],
                results_count=len(results),
                latency_ms=latency_ms,
            )

            return results

        except Exception as e:
            logger.error("document_search_failed", error=str(e), question=question[:100])
            raise RetrievalError(f"Document search failed: {e}") from e

    def _deduplicate(self, results: list[DocumentSearchResult]) -> list[DocumentSearchResult]:
        if len(results) <= 1:
            return results

        dict_results = [
            {"content": r.chunk_content, "metadata": r.metadata, "score": r.score, "document_id": r.document_id, "title": r.title}
            for r in results
        ]

        deduped = self.deduplicator.deduplicate(dict_results, content_key="content")

        return [
            DocumentSearchResult(
                document_id=r["document_id"],
                title=r["title"],
                chunk_content=r["content"][:500],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in deduped
        ]

    async def _rerank(self, question: str, results: list[DocumentSearchResult]) -> list[DocumentSearchResult]:
        dict_results = [
            {"content": r.chunk_content, "metadata": r.metadata, "score": r.score, "document_id": r.document_id, "title": r.title}
            for r in results
        ]

        reranked = self.reranker.rerank(question, dict_results, top_k=self.quality_config.rerank_top_k)

        return [
            DocumentSearchResult(
                document_id=r["document_id"],
                title=r["title"],
                chunk_content=r["content"][:500],
                score=r.get("rerank_score", r["score"]),
                metadata=r["metadata"],
            )
            for r in reranked
        ]

    def get_retrieval_metrics(
        self,
        question: str,
        results: list[DocumentSearchResult],
        latency_ms: float,
    ) -> RetrievalMetrics:
        return RetrievalMetrics(
            query_embedding_dim=self.embedding_service.get_dimension(),
            retrieved_count=len(results),
            reranked=self.quality_config.rerank_enabled,
            top_scores=[r.score for r in results[:5]],
            latency_ms=latency_ms,
            filter_applied=self.config.metadata_filters if self.config.metadata_filters else None,
        )

    def evaluate_confidence(
        self,
        sql_result: Any | None = None,
        doc_results: list[DocumentSearchResult] | None = None,
        financial_result: Any | None = None,
        forecast_result: Any | None = None,
    ) -> tuple[float, str | None]:
        return self.confidence_evaluator.evaluate(
            sql_result,
            [{"score": r.score, "metadata": r.metadata} for r in doc_results] if doc_results else [],
            financial_result,
            forecast_result,
        )