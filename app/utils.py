"""
SAIA-RAG Utility Functions

Common utility functions used across the application.
Follows clean architecture patterns with proper error handling.
"""

import hashlib
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

# Get logger
logger = structlog.get_logger()


def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize dictionary data for safe logging by removing sensitive fields.
    
    Args:
        data: Dictionary to sanitize
        
    Returns:
        Sanitized dictionary safe for logging
    """
    sensitive_keys = {
        'api_key', 'openai_api_key', 'password', 'token', 'secret',
        'authorization', 'auth', 'key', 'credential'
    }
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_logging(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_for_logging(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def clean_text(text: str) -> str:
    """
    Clean and normalize text input.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    return text


def generate_document_id(title: str, content: str) -> str:
    """
    Generate a unique document ID based on title and content.
    
    Args:
        title: Document title
        content: Document content
        
    Returns:
        Unique document ID
    """
    content_hash = hashlib.md5(f"{title}:{content}".encode()).hexdigest()
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"doc_{timestamp}_{content_hash[:8]}"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size - 100:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + chunk_size - overlap, end)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime for consistent display.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted timestamp string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def validate_document_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean document metadata.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Validated metadata
    """
    validated = {}
    
    # Required fields
    if 'title' in metadata:
        validated['title'] = clean_text(str(metadata['title']))[:200]
    
    # Optional fields with defaults
    validated['category'] = clean_text(str(metadata.get('category', 'general')))[:50]
    validated['author'] = clean_text(str(metadata.get('author', 'Unknown')))[:100]
    
    # Tags handling
    tags = metadata.get('tags', [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(',')]
    elif not isinstance(tags, list):
        tags = []
    
    validated['tags'] = [clean_text(tag)[:50] for tag in tags if tag.strip()][:10]
    
    # Timestamps
    validated['upload_date'] = format_timestamp(datetime.utcnow())
    
    return validated
