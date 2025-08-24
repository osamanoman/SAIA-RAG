"""
SAIA-RAG Document Retrieval Module

RAG retrieval logic for finding relevant document chunks.
Handles vector search, context building, and relevance scoring.
"""

from typing import List, Dict, Any, Optional
import structlog

from .config import get_settings
from .vector_store import get_vector_store
from .openai_client import get_openai_client

# Get logger
logger = structlog.get_logger()


class DocumentRetriever:
    """
    Document retrieval service for RAG operations.
    
    Handles:
    1. Query embedding generation
    2. Vector similarity search
    3. Context building and ranking
    4. Relevance filtering
    """
    
    def __init__(self):
        """Initialize the document retriever."""
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.openai_client = get_openai_client()
        
        logger.info("Document retriever initialized")
    
    async def retrieve_relevant_context(
        self,
        query: str,
        max_chunks: int = None,
        confidence_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to retrieve
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Use defaults if not provided
            max_chunks = max_chunks or self.settings.max_search_results
            confidence_threshold = confidence_threshold or self.settings.confidence_threshold
            
            # Generate query embedding
            query_embedding = await self.openai_client.generate_embedding(query)
            
            # Search for similar documents
            search_results = self.vector_store.search_documents(
                query_vector=query_embedding,
                limit=max_chunks,
                score_threshold=confidence_threshold
            )
            
            # Process and rank results
            relevant_chunks = self._process_search_results(search_results)
            
            logger.info(
                "Context retrieval completed",
                query_length=len(query),
                chunks_found=len(relevant_chunks),
                confidence_threshold=confidence_threshold
            )
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(
                "Context retrieval failed",
                query=query[:100],
                error=str(e)
            )
            raise
    
    def _process_search_results(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and enhance search results.
        
        Args:
            search_results: Raw search results from vector store
            
        Returns:
            Processed and enhanced results
        """
        processed_results = []
        
        for result in search_results:
            # Extract metadata
            metadata = result.get("payload", {})
            
            # Build enhanced result
            enhanced_result = {
                "text": metadata.get("text", ""),
                "score": result.get("score", 0.0),
                "document_id": metadata.get("document_id"),
                "title": metadata.get("title", "Untitled"),
                "category": metadata.get("category", "general"),
                "author": metadata.get("author", "Unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "upload_date": metadata.get("upload_date"),
                "tags": metadata.get("tags", [])
            }
            
            processed_results.append(enhanced_result)
        
        # Sort by relevance score (descending)
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        
        return processed_results
    
    def build_context_string(
        self,
        relevant_chunks: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> str:
        """
        Build context string from relevant chunks.
        
        Args:
            relevant_chunks: List of relevant document chunks
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        if not relevant_chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            # Format chunk with metadata
            chunk_text = chunk["text"]
            title = chunk["title"]
            category = chunk["category"]
            
            formatted_chunk = f"[{category}] {title}:\n{chunk_text}\n"
            
            # Check if adding this chunk would exceed limit
            if current_length + len(formatted_chunk) > max_context_length:
                break
            
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        context = "\n---\n".join(context_parts)
        
        logger.debug(
            "Context string built",
            chunks_used=len(context_parts),
            context_length=len(context)
        )
        
        return context
    
    def extract_source_citations(
        self,
        relevant_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract source citations from relevant chunks.
        
        Args:
            relevant_chunks: List of relevant document chunks
            
        Returns:
            List of source citations
        """
        citations = []
        seen_documents = set()
        
        for chunk in relevant_chunks:
            document_id = chunk["document_id"]
            
            # Avoid duplicate citations for the same document
            if document_id not in seen_documents:
                citation = {
                    "document_id": document_id,
                    "title": chunk["title"],
                    "category": chunk["category"],
                    "author": chunk["author"],
                    "upload_date": chunk["upload_date"],
                    "relevance_score": chunk["score"]
                }
                
                citations.append(citation)
                seen_documents.add(document_id)
        
        return citations


# Global instance
_retriever = None


def get_document_retriever() -> DocumentRetriever:
    """
    Get global document retriever instance.
    
    Returns:
        DocumentRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever
