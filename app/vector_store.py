"""
SAIA-RAG Vector Store Module

Qdrant client operations following clean architecture patterns.
Provides global instance pattern for connection pooling and performance.
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
import asyncio
import uuid
import hashlib
from datetime import datetime

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)
import structlog

from .config import get_settings, Settings

# Get logger
logger = structlog.get_logger()


class QdrantVectorStore:
    """
    Qdrant vector store operations with connection pooling and health monitoring.
    
    Follows development rules:
    - Uses qdrant-client directly (no wrapper abstractions)
    - Implements global instance pattern for performance
    - Uses mandatory collection naming: docs_{tenant_id}
    - Uses COSINE distance metric (immutable)
    - Supports both 1536 and 3072 vector dimensions
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Qdrant client with configuration.
        
        Args:
            settings: Application settings with Qdrant configuration
        """
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.get_collection_name()
        
        logger.info(
            "QdrantVectorStore initialized",
            qdrant_url=settings.qdrant_url,
            collection_name=self.collection_name,
            embed_dim=settings.embed_dim
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Qdrant service health and connectivity.
        
        Returns:
            Health status dictionary with connection info
        """
        try:
            start_time = datetime.utcnow()
            
            # Test basic connectivity
            collections = self.client.get_collections()
            
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            health_data = {
                "status": "healthy",
                "response_time_ms": response_time_ms,
                "collections_count": len(collections.collections),
                "url": self.settings.qdrant_url,
                "collection_name": self.collection_name
            }
            
            logger.info("Qdrant health check successful", **health_data)
            return health_data
            
        except Exception as e:
            error_data = {
                "status": "unhealthy",
                "error": str(e),
                "url": self.settings.qdrant_url
            }
            logger.error("Qdrant health check failed", **error_data)
            return error_data
    
    def ensure_collection_exists(self) -> bool:
        """
        Ensure the tenant collection exists, create if not.
        
        Returns:
            True if collection exists or was created successfully
            
        Raises:
            Exception: If collection creation fails
        """
        try:
            # Check if collection exists
            if self.client.collection_exists(self.collection_name):
                logger.info("Collection already exists", collection_name=self.collection_name)
                return True
            
            # Create collection with standard HNSW configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embed_dim,
                    distance=Distance.COSINE  # IMMUTABLE per dev rules
                ),
                # Standard HNSW configuration for production use
                hnsw_config=models.HnswConfigDiff(
                    m=16,  # Standard number of connections per node
                    ef_construct=100,  # Build-time search parameter
                    full_scan_threshold=1000  # Use exact search for small collections
                ),
                # Standard optimizer configuration
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,  # Index when segment reaches threshold
                    max_segment_size=200000,  # Maximum vectors per segment
                    flush_interval_sec=5  # Flush interval in seconds
                )
            )
            
            logger.info(
                "Collection created successfully",
                collection_name=self.collection_name,
                vector_size=self.settings.embed_dim,
                distance_metric="COSINE"
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to ensure collection exists",
                collection_name=self.collection_name,
                error=str(e)
            )
            raise
    
    def upsert_points(
        self, 
        points: List[PointStruct], 
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Upsert points into the collection.
        
        Args:
            points: List of PointStruct objects with id, vector, and payload
            wait: Whether to wait for operation completion
            
        Returns:
            Operation result information
            
        Raises:
            Exception: If upsert operation fails
        """
        try:
            # Ensure collection exists before upserting
            self.ensure_collection_exists()
            
            # Perform upsert operation
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=wait,  # Wait for completion based on parameter
                points=points
            )
            
            logger.info(
                "Points upserted successfully",
                collection_name=self.collection_name,
                points_count=len(points),
                operation_id=operation_info.operation_id if hasattr(operation_info, 'operation_id') else None
            )
            
            return {
                "status": "success",
                "points_count": len(points),
                "collection_name": self.collection_name,
                "operation_info": operation_info
            }
            
        except Exception as e:
            logger.error(
                "Failed to upsert points",
                collection_name=self.collection_name,
                points_count=len(points),
                error=str(e)
            )
            raise
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 8,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            query_filter: Optional filter to apply to search
            with_payload: Whether to include payload in results
            with_vectors: Whether to include vectors in results
            
        Returns:
            List of search results with scores and metadata
            
        Raises:
            Exception: If search operation fails
        """
        try:
            # Ensure collection exists before searching
            self.ensure_collection_exists()
            
            # Perform search operation using HNSW index
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            # Convert results to dictionaries for easier handling
            results = []
            for result in search_results:
                result_dict = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload if with_payload else None,
                    "vector": result.vector if with_vectors else None
                }
                results.append(result_dict)
            
            logger.info(
                "Vector search completed",
                collection_name=self.collection_name,
                query_vector_dim=len(query_vector),
                results_count=len(results),
                limit=limit,
                score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Failed to search similar vectors",
                collection_name=self.collection_name,
                query_vector_dim=len(query_vector),
                limit=limit,
                error=str(e)
            )
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information and statistics
            
        Raises:
            Exception: If collection info retrieval fails
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get point count
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            
            info_dict = {
                "collection_name": self.collection_name,
                "status": collection_info.status,
                "vectors_count": count_result.count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.name,
                },
                "points_count": count_result.count
            }
            
            logger.info("Collection info retrieved", **info_dict)
            return info_dict
            
        except Exception as e:
            logger.error(
                "Failed to get collection info",
                collection_name=self.collection_name,
                error=str(e)
            )
            raise
    
    def delete_points(self, point_ids: List[str]) -> Dict[str, Any]:
        """
        Delete points from the collection by IDs.
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            Deletion operation result
            
        Raises:
            Exception: If deletion fails
        """
        try:
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
                wait=True
            )
            
            result = {
                "status": "success",
                "deleted_count": len(point_ids),
                "collection_name": self.collection_name,
                "operation_info": operation_info
            }
            
            logger.info(
                "Points deleted successfully",
                collection_name=self.collection_name,
                deleted_count=len(point_ids)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to delete points",
                collection_name=self.collection_name,
                point_ids_count=len(point_ids),
                error=str(e)
            )
            raise

    # === DOCUMENT-SPECIFIC OPERATIONS ===

    def index_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Index document chunks with embeddings into the vector store.

        Args:
            document_id: Unique document identifier
            chunks: List of document chunks with metadata
            embeddings: List of embedding vectors for each chunk

        Returns:
            Indexing operation result

        Raises:
            ValueError: If chunks and embeddings length mismatch
            Exception: If indexing operation fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})")

        try:
            # Create points for each chunk
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique point ID for this chunk
                chunk_id = f"{document_id}_chunk_{i}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

                # Create payload with document and chunk metadata
                payload = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "indexed_at": datetime.utcnow().isoformat()
                }

                # Add document-level metadata to each chunk
                chunk_metadata = chunk.get("metadata", {})
                if "title" in chunk_metadata:
                    payload["title"] = chunk_metadata["title"]
                if "category" in chunk_metadata:
                    payload["category"] = chunk_metadata["category"]
                if "tags" in chunk_metadata:
                    payload["tags"] = chunk_metadata["tags"]
                if "author" in chunk_metadata:
                    payload["author"] = chunk_metadata["author"]

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))

            # Upsert all points
            result = self.upsert_points(points)

            logger.info(
                "Document chunks stored successfully",
                collection_name=self.collection_name,
                chunks_count=len(points)
            )

            logger.info(
                "Document chunks indexed successfully",
                document_id=document_id,
                chunks_count=len(chunks),
                collection_name=self.collection_name
            )

            return {
                "status": "success",
                "document_id": document_id,
                "chunks_indexed": len(chunks),
                "collection_name": self.collection_name,
                "operation_result": result
            }

        except Exception as e:
            logger.error(
                "Failed to index document chunks",
                document_id=document_id,
                chunks_count=len(chunks),
                error=str(e)
            )
            raise



    def search_documents(
        self,
        query_vector: List[float],
        limit: int = 8,
        score_threshold: Optional[float] = None,
        document_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using vector similarity.

        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            document_filter: Optional filters (document_id, category, tags)

        Returns:
            List of relevant document chunks with scores and metadata
        """
        try:
            # Build query filter if provided
            query_filter = None
            if document_filter:
                conditions = []

                if "document_id" in document_filter:
                    conditions.append(FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_filter["document_id"])
                    ))

                if "category" in document_filter:
                    conditions.append(FieldCondition(
                        key="category",
                        match=MatchValue(value=document_filter["category"])
                    ))

                if "tags" in document_filter:
                    # Filter by any of the provided tags
                    for tag in document_filter["tags"]:
                        conditions.append(FieldCondition(
                            key="tags",
                            match=MatchValue(value=tag)
                        ))

                if conditions:
                    query_filter = Filter(must=conditions)

            # Perform search
            results = self.search_similar(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False
            )

            # Format results for document context
            formatted_results = []
            for result in results:
                if result["payload"]:
                    formatted_result = {
                        "chunk_id": result["payload"].get("chunk_id"),
                        "document_id": result["payload"].get("document_id"),
                        "text": result["payload"].get("text", ""),
                        "score": result["score"],
                        "chunk_index": result["payload"].get("chunk_index", 0),
                        "title": result["payload"].get("title"),
                        "category": result["payload"].get("category"),
                        "metadata": result["payload"].get("metadata", {}),
                        "indexed_at": result["payload"].get("indexed_at")
                    }
                    formatted_results.append(formatted_result)

            logger.info(
                "Document search completed",
                query_vector_dim=len(query_vector),
                results_count=len(formatted_results),
                limit=limit,
                score_threshold=score_threshold,
                filters=document_filter
            )

            return formatted_results

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                "Failed to search documents",
                query_vector_dim=len(query_vector),
                limit=limit,
                error=str(e),
                error_type=error_type
            )

            # Provide specific error messages for common Qdrant issues
            if "connection" in str(e).lower():
                raise Exception("Cannot connect to vector database. Please check Qdrant service.")
            elif "collection" in str(e).lower():
                raise Exception("Vector collection not found. Please upload documents first.")
            elif "dimension" in str(e).lower():
                raise Exception("Vector dimension mismatch. Please check embedding configuration.")
            else:
                raise Exception(f"Vector search error: {str(e)}")

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document.

        Args:
            document_id: Document identifier to delete

        Returns:
            Deletion operation result
        """
        try:
            # Search for all points with this document_id
            filter_condition = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )

            # Delete points matching the filter
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
                wait=True
            )

            result = {
                "status": "success",
                "document_id": document_id,
                "collection_name": self.collection_name,
                "operation_info": operation_info
            }

            logger.info(
                "Document deleted successfully",
                document_id=document_id,
                collection_name=self.collection_name
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to delete document",
                document_id=document_id,
                collection_name=self.collection_name,
                error=str(e)
            )
            raise

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            List of document chunks with metadata
        """
        try:
            # Search for all chunks of this document
            filter_condition = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )

            # Use scroll to get all points (not limited by search limit)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                with_payload=True,
                with_vectors=False,
                limit=1000  # Reasonable limit for document chunks
            )

            # Format results
            chunks = []
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                if point.payload:
                    chunk = {
                        "chunk_id": point.payload.get("chunk_id"),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "text": point.payload.get("text", ""),
                        "title": point.payload.get("title"),
                        "category": point.payload.get("category"),
                        "metadata": point.payload.get("metadata", {}),
                        "indexed_at": point.payload.get("indexed_at")
                    }
                    chunks.append(chunk)

            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])

            logger.info(
                "Document chunks retrieved",
                document_id=document_id,
                chunks_count=len(chunks)
            )

            return chunks

        except Exception as e:
            logger.error(
                "Failed to get document chunks",
                document_id=document_id,
                error=str(e)
            )
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store with their metadata.

        Returns:
            List of document metadata dictionaries
        """
        try:
            # Scroll through all points to get unique documents
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Large limit to get all documents
                with_payload=True,
                with_vectors=False
            )

            # Group by document_id to get unique documents
            documents_dict = {}
            for point in scroll_result[0]:
                if point.payload:
                    doc_id = point.payload.get("document_id")
                    if doc_id and doc_id not in documents_dict:
                        documents_dict[doc_id] = {
                            "document_id": doc_id,
                            "title": point.payload.get("title"),
                            "category": point.payload.get("category"),
                            "tags": point.payload.get("tags", []),
                            "author": point.payload.get("author") or point.payload.get("metadata", {}).get("author"),
                            "upload_date": point.payload.get("indexed_at"),
                            "chunk_count": 0
                        }

                    # Count chunks for this document
                    if doc_id in documents_dict:
                        documents_dict[doc_id]["chunk_count"] += 1

            # Convert to list and sort by upload date (newest first)
            documents = list(documents_dict.values())
            documents.sort(key=lambda x: x.get("upload_date", ""), reverse=True)

            logger.info(
                "Documents listed",
                total_documents=len(documents),
                collection_name=self.collection_name
            )

            return documents

        except Exception as e:
            logger.error(
                "Failed to list documents",
                error=str(e)
            )
            raise


@lru_cache()
def get_vector_store() -> QdrantVectorStore:
    """
    Get cached vector store instance.
    
    Uses @lru_cache to ensure vector store is initialized only once
    and reused throughout the application lifecycle for optimal performance.
    
    Returns:
        Cached QdrantVectorStore instance
    """
    settings = get_settings()
    return QdrantVectorStore(settings)


# Export for easy importing
__all__ = ["QdrantVectorStore", "get_vector_store"]
