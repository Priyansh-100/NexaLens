import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.rag.services.rag_retriever import RAGRetriever
from nexalens.rag.schemas import DocumentSearchResult, RAGRetrieverConfig


class TestRAGRetriever:
    @pytest.fixture
    def retriever(self):
        return RAGRetriever()

    @pytest.mark.asyncio
    async def test_search_documents(self, retriever):
        mock_chunks = [
            {
                "content": "Revenue was $1M in Q3",
                "metadata": {"document_id": str(uuid4()), "document_title": "Q3 Report", "source_id": str(uuid4())},
                "score": 0.9,
            },
            {
                "content": "Revenue grew 10% YoY",
                "metadata": {"document_id": str(uuid4()), "document_title": "Annual Report", "source_id": str(uuid4())},
                "score": 0.8,
            },
        ]

        with patch.object(retriever.document_service, "search_documents", new=AsyncMock(return_value=mock_chunks)):
            results = await retriever.search_documents("What was Q3 revenue?")
            assert len(results) == 2
            assert all(isinstance(r, DocumentSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, retriever):
        mock_chunks = [
            {
                "content": "High relevance",
                "metadata": {"document_id": str(uuid4()), "document_title": "Doc1"},
                "score": 0.9,
            },
            {
                "content": "Low relevance",
                "metadata": {"document_id": str(uuid4()), "document_title": "Doc2"},
                "score": 0.2,
            },
        ]

        retriever.quality_config.min_score_threshold = 0.5

        with patch.object(retriever.document_service, "search_documents", new=AsyncMock(return_value=mock_chunks)):
            results = await retriever.search_documents("test query")
            assert len(results) == 1
            assert results[0].score >= 0.5

    @pytest.mark.asyncio
    async def test_deduplication(self, retriever):
        mock_chunks = [
            {
                "content": "Revenue was $1M in Q3",
                "metadata": {"document_id": str(uuid4()), "document_title": "Q3 Report"},
                "score": 0.9,
            },
            {
                "content": "Revenue was 1 million in Q3",
                "metadata": {"document_id": str(uuid4()), "document_title": "Q3 Report v2"},
                "score": 0.85,
            },
        ]

        retriever.quality_config.dedup_threshold = 0.8

        with patch.object(retriever.document_service, "search_documents", new=AsyncMock(return_value=mock_chunks)):
            results = await retriever.search_documents("What was Q3 revenue?")
            assert len(results) == 1  # Should deduplicate