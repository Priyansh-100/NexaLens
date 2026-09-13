import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.documents.processor import DocumentProcessor, ProcessedDocument


@pytest.fixture
def doc_processor():
    return DocumentProcessor()


@pytest.fixture
def sample_text():
    return "This is a test document. " * 50


class TestDocumentProcessor:
    def test_validate_file_pdf(self, doc_processor):
        doc_processor.validate_file("test.pdf", b"content")

    def test_validate_file_txt(self, doc_processor):
        doc_processor.validate_file("test.txt", b"content")

    def test_validate_file_unsupported(self, doc_processor):
        with pytest.raises(Exception):
            doc_processor.validate_file("test.exe", b"content")

    def test_validate_file_too_large(self, doc_processor):
        large_content = b"x" * (60 * 1024 * 1024)
        with pytest.raises(Exception):
            doc_processor.validate_file("test.pdf", large_content)

    def test_extract_text_txt(self, doc_processor):
        content = b"Hello world"
        text = doc_processor.extract_text("test.txt", content)
        assert text == "Hello world"

    def test_extract_text_md(self, doc_processor):
        content = b"# Header\n\nContent"
        text = doc_processor.extract_text("test.md", content)
        assert "Header" in text

    @patch("nexalens.documents.processor.pdfplumber.open")
    def test_extract_text_pdf(self, mock_open, doc_processor):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 content"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        text = doc_processor.extract_text("test.pdf", b"fake pdf")
        assert "Page 1 content" in text

    def test_process_document(self, doc_processor, sample_text):
        source_id = uuid4()
        result = doc_processor.process(
            source_id=source_id,
            filename="test.txt",
            content=sample_text.encode(),
            metadata={"author": "test"},
        )

        assert isinstance(result, ProcessedDocument)
        assert result.document.source_id == source_id
        assert result.document.title == "test"
        assert result.document.chunk_count > 0
        assert len(result.chunks) == result.document.chunk_count
        assert result.document.metadata["author"] == "test"
        assert "content_hash" in result.document.metadata

    def test_create_chunks(self, doc_processor):
        from nexalens.models.schemas import Document
        doc = Document(
            id=uuid4(),
            source_id=uuid4(),
            title="Test",
            content="Chunk 1. " * 100 + "Chunk 2. " * 100,
        )

        chunks = doc_processor._create_chunks(doc)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == doc.id

    @pytest.mark.asyncio
    async def test_embed_chunks(self, doc_processor):
        from nexalens.models.schemas import DocumentChunk
        chunks = [
            DocumentChunk(document_id=uuid4(), content=f"Chunk {i}", chunk_index=i)
            for i in range(3)
        ]

        with patch("nexalens.documents.processor.embedding_service") as mock_embed:
            mock_embed.embed_batch = AsyncMock(return_value=[[0.1]*384, [0.2]*384, [0.3]*384])

            result = await doc_processor.embed_chunks(chunks)
            assert all(c.embedding is not None for c in result)
            assert len(result[0].embedding) == 384