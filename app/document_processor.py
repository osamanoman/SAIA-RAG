"""
SAIA-RAG Document Processing Module

Document text extraction, chunking, and preprocessing for RAG system.
Handles various document formats and prepares text for embedding generation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

import structlog

from .config import get_settings

# Get logger
logger = structlog.get_logger()


class DocumentProcessor:
    """
    Document processing for RAG system.
    
    Handles:
    - Text extraction from various formats
    - Intelligent text chunking with overlap
    - Metadata preservation and enhancement
    - Text cleaning and preprocessing
    """
    
    def __init__(self, settings=None):
        """Initialize document processor with configuration."""
        self.settings = settings or get_settings()
        
        # Chunking configuration
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        self.min_chunk_size = 100  # Minimum viable chunk size
        
        logger.info(
            "Document processor initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size
        )
    
    def extract_text_from_content(
        self, 
        content: str, 
        content_type: str = "text/plain"
    ) -> str:
        """
        Extract text from content based on content type.
        
        Args:
            content: Raw content string
            content_type: MIME type of content
            
        Returns:
            Extracted plain text
        """
        try:
            if content_type == "text/plain":
                return content
            elif content_type == "text/html":
                # Basic HTML tag removal (would use BeautifulSoup in production)
                text = re.sub(r'<[^>]+>', '', content)
                text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # Remove HTML entities
                return text
            elif content_type == "text/markdown":
                # Basic markdown cleanup (would use markdown parser in production)
                text = re.sub(r'#{1,6}\s+', '', content)  # Remove headers
                text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
                text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
                text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
                return text
            else:
                # For unsupported types, return as-is
                logger.warning(f"Unsupported content type: {content_type}")
                return content
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return content
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunking boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (would use spaCy or NLTK in production)
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks = []
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_data = {
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "char_count": len(current_chunk.strip()),
                    "metadata": metadata or {}
                }
                
                # Add chunk hash for deduplication
                chunk_data["content_hash"] = hashlib.md5(
                    chunk_data["text"].encode()
                ).hexdigest()
                
                chunks.append(chunk_data)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length = len(current_chunk)
        
        # Add final chunk if it has content
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_data = {
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "char_count": len(current_chunk.strip()),
                "metadata": metadata or {}
            }
            
            chunk_data["content_hash"] = hashlib.md5(
                chunk_data["text"].encode()
            ).hexdigest()
            
            chunks.append(chunk_data)
        
        logger.info(
            "Text chunking completed",
            original_length=len(text),
            chunks_created=len(chunks),
            avg_chunk_size=sum(c["char_count"] for c in chunks) // len(chunks) if chunks else 0
        )
        
        return chunks
    
    def process_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """
        Process a complete document: extract text, clean, and chunk.
        
        Args:
            content: Raw document content
            metadata: Document metadata
            content_type: MIME type of content
            
        Returns:
            Processing result with chunks and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Extract text from content
            raw_text = self.extract_text_from_content(content, content_type)
            
            # Clean text
            clean_text = self.clean_text(raw_text)
            
            if not clean_text:
                return {
                    "success": False,
                    "error": "No text content found after processing",
                    "chunks": []
                }
            
            # Create chunks
            chunks = self.create_chunks(clean_text, metadata)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No valid chunks created from document",
                    "chunks": []
                }
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = {
                "success": True,
                "chunks": chunks,
                "original_length": len(content),
                "processed_length": len(clean_text),
                "chunks_count": len(chunks),
                "processing_time_ms": processing_time_ms,
                "content_type": content_type
            }
            
            logger.info(
                "Document processing completed",
                **{k: v for k, v in result.items() if k != "chunks"}
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Document processing failed",
                content_length=len(content),
                content_type=content_type,
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "chunks": []
            }


def get_document_processor() -> DocumentProcessor:
    """
    Get document processor instance.
    
    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor()


# Export for easy importing
__all__ = ["DocumentProcessor", "get_document_processor"]
