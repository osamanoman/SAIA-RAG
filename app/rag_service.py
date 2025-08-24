"""
SAIA-RAG Service Module

Complete RAG (Retrieval-Augmented Generation) implementation.
Orchestrates document processing, vector search, and response generation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import hashlib
from functools import lru_cache

import structlog

from .config import get_settings
from .openai_client import get_openai_client
from .vector_store import get_vector_store
from .document_processor import get_document_processor
from .models import SourceDocument

# Get logger
logger = structlog.get_logger()


class RAGService:
    """
    Complete RAG service implementation.
    
    Provides:
    - Document ingestion with embedding generation
    - Context retrieval from vector store
    - Response generation with source citations
    - Conversation context management
    """
    
    def __init__(self):
        """Initialize RAG service with dependencies."""
        self.settings = get_settings()
        self.openai_client = get_openai_client()
        self.vector_store = get_vector_store()
        self.document_processor = get_document_processor()

        # Simple in-memory cache for frequent queries
        self._query_cache = {}
        self._cache_max_size = 100

        logger.info("RAG service initialized", cache_max_size=self._cache_max_size)
    
    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document content
            metadata: Document metadata
            content_type: MIME type of content
            
        Returns:
            Ingestion result with document ID and chunk count
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Process document into chunks
            processing_result = self.document_processor.process_document(
                content, metadata, content_type
            )
            
            if not processing_result["success"]:
                return {
                    "success": False,
                    "error": processing_result["error"],
                    "document_id": document_id
                }
            
            chunks = processing_result["chunks"]
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.openai_client.generate_embeddings(chunk_texts)
            
            # Index chunks in vector store
            index_result = self.vector_store.index_document_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings
            )
            
            end_time = datetime.utcnow()
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(chunks),
                "chunks_indexed": index_result["chunks_indexed"],
                "processing_time_ms": processing_result["processing_time_ms"],
                "total_time_ms": total_time_ms,
                "metadata": metadata
            }
            
            logger.info(
                "Document ingestion completed",
                document_id=document_id,
                chunks_created=len(chunks),
                total_time_ms=total_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Document ingestion failed",
                error=str(e),
                error_type=type(e).__name__,
                metadata=metadata
            )

            # Provide specific error messages based on error type
            error_message = str(e)
            if "openai" in str(e).lower():
                error_message = "Failed to generate embeddings. Please check OpenAI API configuration."
            elif "qdrant" in str(e).lower():
                error_message = "Failed to index document in vector database. Please try again."
            elif "processing" in str(e).lower():
                error_message = "Failed to process document content. Please check document format."

            return {
                "success": False,
                "error": error_message,
                "document_id": document_id if 'document_id' in locals() else None,
                "error_type": type(e).__name__
            }
    
    def _get_cache_key(self, query: str, max_context_chunks: int, confidence_threshold: float) -> str:
        """Generate cache key for query."""
        cache_data = f"{query.lower().strip()}_{max_context_chunks}_{confidence_threshold}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        if cache_key in self._query_cache:
            cached_data = self._query_cache[cache_key]
            # Simple cache expiry - 5 minutes
            if (datetime.utcnow() - cached_data["timestamp"]).seconds < 300:
                logger.info("Cache hit", cache_key=cache_key[:8])
                return cached_data["response"]
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response with timestamp."""
        # Implement simple LRU by removing oldest entries
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = min(self._query_cache.keys(),
                           key=lambda k: self._query_cache[k]["timestamp"])
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = {
            "response": response,
            "timestamp": datetime.utcnow()
        }
        logger.info("Response cached", cache_key=cache_key[:8])

    async def generate_response(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        max_context_chunks: int = 8,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate RAG response for a user query.

        Args:
            query: User query
            conversation_id: Optional conversation identifier
            max_context_chunks: Maximum chunks to use for context
            confidence_threshold: Minimum relevance score for chunks

        Returns:
            Generated response with sources and metadata
        """
        try:
            start_time = datetime.utcnow()
            confidence_threshold = confidence_threshold or self.settings.confidence_threshold

            # Check cache first
            cache_key = self._get_cache_key(query, max_context_chunks, confidence_threshold)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                # Update conversation_id and return cached response
                cached_response["conversation_id"] = conversation_id
                return cached_response
            
            # Generate query embedding
            query_embedding = await self.openai_client.generate_embedding(query)
            
            # Search for relevant chunks
            confidence_threshold = confidence_threshold or self.settings.confidence_threshold
            
            search_results = self.vector_store.search_documents(
                query_vector=query_embedding,
                limit=max_context_chunks,
                score_threshold=confidence_threshold
            )
            
            # Build context from search results
            context_chunks = []
            sources = []

            logger.info("Processing search results", results_count=len(search_results))

            for i, result in enumerate(search_results):
                try:
                    context_chunks.append(result["text"])

                    # Create source document reference with validation
                    source = SourceDocument(
                        document_id=result["document_id"],
                        chunk_id=result["chunk_id"],
                        title=result.get("title"),
                        relevance_score=float(result["score"]),  # Ensure it's a float
                        text_excerpt=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                    )
                    sources.append(source)
                    logger.info(f"Created source {i}", source_type=type(source))

                except Exception as e:
                    logger.error(f"Failed to create source {i}", error=str(e), result_keys=list(result.keys()))
                    continue
            
            # Build context string
            context = "\n\n".join([
                f"Source {i+1}: {chunk}" 
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Create system prompt for RAG
            system_prompt = self._build_system_prompt(context)
            
            # Generate response using OpenAI
            messages = [
                {"role": "user", "content": query}
            ]
            
            chat_result = await self.openai_client.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            # Calculate confidence based on source relevance
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources) if sources else 0.0
            confidence = min(avg_relevance * 1.2, 1.0)  # Boost confidence slightly
            
            end_time = datetime.utcnow()
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Validate and create result with proper types
            result = {
                "response": str(chat_result["content"]) if chat_result["content"] else "I apologize, but I couldn't generate a response.",
                "confidence": float(confidence),
                "sources": sources,  # List of SourceDocument objects
                "sources_count": int(len(sources)),
                "conversation_id": conversation_id,
                "processing_time_ms": int(total_time_ms),
                "tokens_used": int(chat_result.get("tokens_used", 0)),
                "model_used": str(chat_result.get("model", "unknown"))
            }

            logger.info("RAG result created",
                       result_keys=list(result.keys()),
                       sources_type=type(result["sources"]),
                       sources_count=len(result["sources"]))
            
            logger.info(
                "RAG response generated",
                query_length=len(query),
                sources_count=len(sources),
                confidence=confidence,
                processing_time_ms=total_time_ms,
                tokens_used=chat_result["tokens_used"]
            )

            # Cache the response for future use
            self._cache_response(cache_key, result)

            return result
            
        except Exception as e:
            logger.error(
                "RAG response generation failed",
                query=query[:100],
                conversation_id=conversation_id,
                error=str(e),
                error_type=type(e).__name__
            )

            # Return a graceful fallback response instead of crashing
            return {
                "response": "I apologize, but I'm experiencing technical difficulties processing your request. Please try again later or contact support if the issue persists.",
                "confidence": 0.0,
                "sources": [],
                "sources_count": 0,
                "conversation_id": conversation_id,
                "processing_time_ms": 0,
                "tokens_used": 0,
                "model_used": "fallback",
                "error": str(e)
            }
    
    def _build_system_prompt(self, context: str) -> str:
        """
        Build system prompt for RAG response generation.
        
        Args:
            context: Retrieved context from vector search
            
        Returns:
            System prompt string
        """
        return f"""You are SAIA, a helpful AI assistant for customer support. Your role is to provide accurate, helpful responses based on the provided context.

CONTEXT INFORMATION:
{context}

INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your response
4. If you reference specific information, you can mention it comes from the provided sources
5. Maintain a helpful, professional tone
6. If the user asks about something not covered in the context, politely explain that you don't have that information available

Remember: Only use the context provided above to answer questions. Do not use external knowledge beyond what's given in the context."""
    
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search documents using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_score: Minimum relevance score
            filters: Optional search filters
            
        Returns:
            Search results with relevance scores
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate query embedding
            query_embedding = await self.openai_client.generate_embedding(query)
            
            # Perform vector search
            search_results = self.vector_store.search_documents(
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
                document_filter=filters
            )
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "chunk_id": result["chunk_id"],
                    "document_id": result["document_id"],
                    "title": result.get("title"),
                    "content": result["text"],
                    "score": result["score"],
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)
            
            result = {
                "results": formatted_results,
                "total_results": len(formatted_results),
                "processing_time_ms": processing_time_ms,
                "query": query
            }
            
            logger.info(
                "Document search completed",
                query=query[:100],
                results_count=len(formatted_results),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Document search failed",
                query=query[:100],
                error=str(e)
            )
            raise


def get_rag_service() -> RAGService:
    """
    Get RAG service instance.
    
    Returns:
        RAG service instance
    """
    return RAGService()


# Export for easy importing
__all__ = ["RAGService", "get_rag_service"]
