"""
SAIA-RAG Vector Store Module

Qdrant client operations following clean architecture patterns.
Provides global instance pattern for connection pooling and performance.
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from qdrant_client import QdrantClient
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
            
            # Create collection with mandatory configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embed_dim,
                    distance=Distance.COSINE  # IMMUTABLE per dev rules
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
                wait=wait,
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
            
            # Perform search operation
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
