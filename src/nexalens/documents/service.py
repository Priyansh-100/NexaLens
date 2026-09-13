from typing import Any
from uuid import UUID

from nexalens.core.logging import get_logger
from nexalens.documents.processor import document_processor, ProcessedDocument
from nexalens.models.database import DocumentModel, DocumentChunkModel
from nexalens.models.schemas import Document, DocumentChunk
from nexalens.services.vector_store import vector_store

logger = get_logger(__name__)


class DocumentService:
    async def ingest_document(
        self,
        session,
        source_id: UUID,
        filename: str,
        content: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        processed = document_processor.process(source_id, filename, content, metadata)
        processed.chunks = await document_processor.embed_chunks(processed.chunks)

        doc_model = DocumentModel(
            id=processed.document.id,
            source_id=processed.document.source_id,
            title=processed.document.title,
            content=processed.document.content,
            metadata=processed.document.metadata,
            chunk_count=len(processed.chunks),
        )
        session.add(doc_model)

        for chunk in processed.chunks:
            chunk_model = DocumentChunkModel(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
                embedding=chunk.embedding,
            )
            session.add(chunk_model)

        await session.flush()
        await vector_store.add_chunks(processed.chunks)

        logger.info("document_ingested", document_id=str(processed.document.id), chunks=len(processed.chunks))
        return processed.document

    async def delete_document(self, session, document_id: UUID) -> bool:
        doc = await session.get(DocumentModel, document_id)
        if not doc:
            return False

        await session.delete(doc)
        await vector_store.delete_document_chunks(document_id)
        logger.info("document_deleted", document_id=str(document_id))
        return True

    async def search_documents(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_ids: list[UUID] | None = None,
    ) -> list[dict[str, Any]]:
        filter_dict = None
        if source_ids:
            filter_dict = {"source_id": {"$in": [str(sid) for sid in source_ids]}}

        return await vector_store.search(query_embedding, top_k=top_k, filter=filter_dict)


document_service = DocumentService()