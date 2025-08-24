"""
SAIA-RAG Document Ingestion Module

Document processing and ingestion functionality.
Handles text extraction, chunking, embedding generation, and vector storage.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from .config import get_settings
from .vector_store import get_vector_store
from .openai_client import get_openai_client
from .utils import chunk_text, generate_document_id, validate_document_metadata

# Get logger
logger = structlog.get_logger()


class DocumentIngestor:
    """
    Document ingestion service for processing and storing documents.
    
    Handles the complete ingestion pipeline:
    1. Text processing and cleaning
    2. Intelligent chunking with overlap
    3. Embedding generation via OpenAI
    4. Vector storage in Qdrant
    """
    
    def __init__(self):
        """Initialize the document ingestor."""
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.openai_client = get_openai_client()
        
        logger.info("Document ingestor initialized")
    
    async def ingest_document(
        self,
        title: str,
        content: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            title: Document title
            content: Document content
            category: Document category
            tags: Document tags
            author: Document author
            
        Returns:
            Ingestion result with document ID and chunk count
        """
        try:
            # Validate and prepare metadata
            metadata = validate_document_metadata({
                'title': title,
                'category': category or 'general',
                'tags': tags or [],
                'author': author or 'Unknown'
            })
            
            # Generate unique document ID
            document_id = generate_document_id(title, content)
            
            # Process document content
            chunks = await self._process_document_content(content)
            
            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings(chunks)
            
            # Store in vector database
            await self._store_document_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(
                "Document ingested successfully",
                document_id=document_id,
                title=title,
                chunks_count=len(chunks)
            )
            
            return {
                "document_id": document_id,
                "title": title,
                "chunks_created": len(chunks),
                "status": "success",
                "upload_date": metadata["upload_date"]
            }
            
        except Exception as e:
            logger.error(
                "Document ingestion failed",
                title=title,
                error=str(e)
            )
            raise
    
    async def _process_document_content(self, content: str) -> List[str]:
        """
        Process document content into chunks.
        
        Args:
            content: Raw document content
            
        Returns:
            List of processed text chunks
        """
        # Clean and normalize content
        content = content.strip()
        if not content:
            raise ValueError("Document content cannot be empty")
        
        # Split into chunks with overlap
        chunks = chunk_text(
            text=content,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap
        )
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 50]
        
        if not chunks:
            raise ValueError("No valid chunks generated from document content")
        
        logger.info(
            "Document content processed",
            original_length=len(content),
            chunks_count=len(chunks)
        )
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = await self.openai_client.generate_embedding(chunk)
                embeddings.append(embedding)
                
                logger.debug(
                    "Embedding generated",
                    chunk_index=i,
                    chunk_length=len(chunk),
                    embedding_dim=len(embedding)
                )
                
            except Exception as e:
                logger.error(
                    "Embedding generation failed",
                    chunk_index=i,
                    error=str(e)
                )
                raise
        
        logger.info(
            "All embeddings generated",
            chunks_count=len(chunks),
            embeddings_count=len(embeddings)
        )
        
        return embeddings
    
    async def _store_document_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store document chunks in vector database.
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Document metadata
        """
        # Prepare chunk data for storage
        chunk_data = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                **metadata,
                "document_id": document_id,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "text": chunk,
                "text_length": len(chunk)
            }
            
            chunk_data.append({
                "text": chunk,
                "embedding": embedding,
                "metadata": chunk_metadata
            })
        
        # Store in vector database
        await self.vector_store.index_document_chunks(
            document_id=document_id,
            chunks=chunk_data
        )
        
        logger.info(
            "Document chunks stored",
            document_id=document_id,
            chunks_stored=len(chunk_data)
        )


# Global instance
_ingestor = None


def get_document_ingestor() -> DocumentIngestor:
    """
    Get global document ingestor instance.
    
    Returns:
        DocumentIngestor instance
    """
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor
