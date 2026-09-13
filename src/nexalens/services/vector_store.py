from typing import Any
from uuid import UUID

import chromadb
from chromadb.config import Settings as ChromaSettings

from nexalens.core.config import get_settings
from nexalens.core.exceptions import VectorStoreError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import DocumentChunk

logger = get_logger(__name__)
settings = get_settings()


class VectorStoreService:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ):
        self.client = chromadb.HttpClient(
            host=host or settings.chroma_host,
            port=port or settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name or settings.chroma_collection
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        try:
            ids = [str(chunk.id) for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks] if chunks[0].embedding else None
            metadatas = [
                {
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }
                for chunk in chunks
            ]

            if embeddings:
                self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            else:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

            logger.info("chunks_added", count=len(chunks))
        except Exception as e:
            logger.error("add_chunks_failed", error=str(e))
            raise VectorStoreError(f"Failed to add chunks: {e}") from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter,
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            chunks = []
            for i, chunk_id in enumerate(results["ids"][0]):
                chunks.append(
                    {
                        "id": UUID(chunk_id),
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i],
                    }
                )
            return chunks
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            raise VectorStoreError(f"Failed to search vectors: {e}") from e

    async def delete_document_chunks(self, document_id: UUID) -> None:
        try:
            self.collection.delete(where={"document_id": str(document_id)})
            logger.info("document_chunks_deleted", document_id=str(document_id))
        except Exception as e:
            logger.error("delete_chunks_failed", error=str(e), document_id=str(document_id))
            raise VectorStoreError(f"Failed to delete chunks: {e}") from e

    async def get_collection_stats(self) -> dict[str, Any]:
        try:
            count = self.collection.count()
            return {"collection": self.collection_name, "chunk_count": count}
        except Exception as e:
            logger.error("stats_failed", error=str(e))
            raise VectorStoreError(f"Failed to get stats: {e}") from e

    async def health_check(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False


vector_store = VectorStoreService()