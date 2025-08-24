"""
SAIA-RAG Pydantic v2 Models

All request/response models for the SAIA-RAG API.
Follows development rules: all models in this file, Pydantic v2 syntax.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# === ENUMS ===

class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class EscalationReason(str, Enum):
    """Escalation reason categories."""
    COMPLEX_TECHNICAL = "complex_technical_issue"
    UNSATISFIED_RESPONSE = "unsatisfied_response"
    HUMAN_REQUESTED = "human_requested"
    SYSTEM_ERROR = "system_error"


class FeedbackCategory(str, Enum):
    """Feedback categories."""
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


# === BASE MODELS ===

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    status: str = Field(..., description="Response status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        use_enum_values=True
    )


class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: str = Field(default="error", description="Error status")
    error: ErrorDetail = Field(..., description="Error details")


# === DOCUMENT MODELS ===

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    title: Optional[str] = Field(None, max_length=200, description="Document title")
    category: Optional[str] = Field(None, max_length=50, description="Document category")
    tags: Optional[List[str]] = Field(default=[], description="Document tags")
    author: Optional[str] = Field(None, max_length=100, description="Document author")
    source: Optional[str] = Field(None, max_length=200, description="Document source")
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate tags list."""
        if v is None:
            return []
        # Remove empty tags and limit to 10 tags
        clean_tags = [tag.strip() for tag in v if tag.strip()]
        return clean_tags[:10]


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metadata": {
                    "title": "User Manual",
                    "category": "documentation",
                    "tags": ["manual", "help", "guide"],
                    "author": "Support Team",
                    "source": "internal"
                }
            }
        }
    )


class DocumentContentRequest(BaseModel):
    """Document content upload request model."""
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    category: Optional[str] = Field(None, max_length=50, description="Document category")
    content: str = Field(..., min_length=1, description="Document content")
    tags: Optional[List[str]] = Field(default=[], description="Document tags")
    author: Optional[str] = Field(None, max_length=100, description="Document author")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate tags list."""
        if v is None:
            return []
        # Remove empty tags and limit to 10 tags
        clean_tags = [tag.strip() for tag in v if tag.strip()]
        return clean_tags[:10]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Password Reset Guide",
                "category": "Support",
                "content": "Step 1: Go to login page...",
                "tags": ["password", "reset", "guide"],
                "author": "Support Team"
            }
        }
    )


class DocumentChunk(BaseModel):
    """Document chunk model."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_index: int = Field(..., ge=0, description="Chunk index within document")
    text: str = Field(..., min_length=1, description="Chunk text content")
    title: Optional[str] = Field(None, description="Chunk title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")


class DocumentResponse(BaseResponse):
    """Document response model."""
    status: str = Field(default="success", description="Success status")
    document_id: str = Field(..., description="Unique document identifier")
    title: Optional[str] = Field(None, description="Document title")
    category: Optional[str] = Field(None, description="Document category")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    upload_date: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")


class DocumentListItem(BaseModel):
    """Document list item model."""
    document_id: str = Field(..., description="Document identifier")
    title: Optional[str] = Field(None, description="Document title")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default=[], description="Document tags")
    chunk_count: int = Field(..., ge=0, description="Number of chunks")
    status: DocumentStatus = Field(..., description="Document status")
    upload_date: datetime = Field(..., description="Upload date")
    author: Optional[str] = Field(None, description="Document author")


class DocumentListResponse(BaseResponse):
    """Document list response model."""
    status: str = Field(default="success", description="Success status")
    documents: List[DocumentListItem] = Field(..., description="List of documents")
    total: int = Field(..., ge=0, description="Total number of documents")
    limit: int = Field(..., ge=1, description="Results limit")
    offset: int = Field(..., ge=0, description="Results offset")


class DocumentDeleteResponse(BaseResponse):
    """Document deletion response model."""
    status: str = Field(default="success", description="Success status")
    document_id: str = Field(..., description="Deleted document ID")
    chunks_deleted: int = Field(..., ge=0, description="Number of chunks deleted")


# === CHAT/RAG MODELS ===

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, max_length=100, description="Conversation identifier")
    max_tokens: Optional[int] = Field(default=500, ge=1, le=2000, description="Maximum response tokens")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and clean message."""
        if not v.strip():
            raise ValueError("Message cannot be empty")

        # Basic sanitization - remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', v.strip())

        return cleaned

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate conversation ID format."""
        if v is None:
            return None

        # Basic alphanumeric validation
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Conversation ID must contain only alphanumeric characters, underscores, and hyphens")

        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "How do I reset my password?",
                "conversation_id": "conv_123",
                "include_sources": True
            }
        }
    )


class SourceDocument(BaseModel):
    """Source document reference model."""
    document_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Source chunk ID")
    title: Optional[str] = Field(None, description="Document title")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    text_excerpt: Optional[str] = Field(None, max_length=200, description="Relevant text excerpt")

    @field_validator("relevance_score")
    @classmethod
    def validate_relevance_score(cls, v: float) -> float:
        """Ensure relevance score is a valid float between 0 and 1."""
        if not isinstance(v, (int, float)):
            raise ValueError("Relevance score must be a number")
        return float(max(0.0, min(1.0, v)))

    @field_validator("text_excerpt")
    @classmethod
    def validate_text_excerpt(cls, v: Optional[str]) -> Optional[str]:
        """Validate and truncate text excerpt."""
        if v is None:
            return None
        if len(v) > 200:
            return v[:197] + "..."
        return v


class ChatResponse(BaseResponse):
    """Chat response model."""
    status: str = Field(default="success", description="Success status")
    response: str = Field(..., description="AI assistant response")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    sources: List[SourceDocument] = Field(default=[], description="Source documents used")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens used in generation")


# === SEARCH MODELS ===

class SearchRequest(BaseModel):
    """Document search request model."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean search query."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """Search result model."""
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseResponse):
    """Search response model."""
    status: str = Field(default="success", description="Success status")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    query: str = Field(..., description="Original search query")


# === ESCALATION MODELS ===

class EscalationRequest(BaseModel):
    """Escalation request model."""
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    reason: EscalationReason = Field(..., description="Escalation reason")
    priority: str = Field(default="medium", description="Escalation priority")
    user_contact: Optional[str] = Field(None, max_length=200, description="User contact information")
    description: Optional[str] = Field(None, max_length=1000, description="Additional description")
    
    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority level."""
        allowed = {"low", "medium", "high", "urgent"}
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of: {allowed}")
        return v.lower()


class EscalationResponse(BaseResponse):
    """Escalation response model."""
    status: str = Field(default="escalated", description="Escalation status")
    escalation_id: str = Field(..., description="Unique escalation identifier")
    ticket_number: Optional[str] = Field(None, description="Support ticket number")
    estimated_response_time: Optional[str] = Field(None, description="Estimated response time")


# === FEEDBACK MODELS ===

class FeedbackRequest(BaseModel):
    """Feedback request model."""
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    message_id: Optional[str] = Field(None, description="Specific message identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback: Optional[str] = Field(None, max_length=1000, description="Feedback text")
    category: FeedbackCategory = Field(..., description="Feedback category")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "conv_123",
                "rating": 4,
                "feedback": "Helpful response, but could be more detailed",
                "category": "helpfulness"
            }
        }
    )


class FeedbackResponse(BaseResponse):
    """Feedback response model."""
    status: str = Field(default="received", description="Feedback status")
    feedback_id: str = Field(..., description="Unique feedback identifier")
    message: str = Field(default="Thank you for your feedback", description="Confirmation message")


# === ADMIN MODELS ===

class SystemStats(BaseModel):
    """System statistics model."""
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    total_chunks: int = Field(..., ge=0, description="Total number of document chunks")
    total_conversations: int = Field(..., ge=0, description="Total number of conversations")
    total_escalations: int = Field(..., ge=0, description="Total number of escalations")
    total_feedback: int = Field(..., ge=0, description="Total feedback submissions")
    avg_response_time_ms: float = Field(..., ge=0, description="Average response time in milliseconds")
    uptime_seconds: int = Field(..., ge=0, description="System uptime in seconds")


class VectorStoreStats(BaseModel):
    """Vector store statistics model."""
    collection_name: str = Field(..., description="Collection name")
    total_points: int = Field(..., ge=0, description="Total points in collection")
    vector_size: int = Field(..., ge=0, description="Vector dimension size")
    distance_metric: str = Field(..., description="Distance metric used")
    index_status: str = Field(..., description="Index status")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage in MB")


class AdminStatsResponse(BaseResponse):
    """Admin statistics response model."""
    status: str = Field(default="success", description="Success status")
    system: SystemStats = Field(..., description="System statistics")
    vector_store: VectorStoreStats = Field(..., description="Vector store statistics")
    collection_time: datetime = Field(default_factory=datetime.utcnow, description="Statistics collection time")


class SystemHealthDetail(BaseModel):
    """Detailed system health model."""
    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    response_time_ms: Optional[int] = Field(None, ge=0, description="Response time in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional component details")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check time")


class AdminHealthResponse(BaseResponse):
    """Admin health response model."""
    status: str = Field(default="success", description="Overall system status")
    components: List[SystemHealthDetail] = Field(..., description="Component health details")
    overall_health: str = Field(..., description="Overall health status")
    checks_performed: int = Field(..., ge=0, description="Number of health checks performed")


class SystemConfigItem(BaseModel):
    """System configuration item model."""
    key: str = Field(..., description="Configuration key")
    value: str = Field(..., description="Configuration value (sanitized)")
    category: str = Field(..., description="Configuration category")
    is_sensitive: bool = Field(default=False, description="Whether the value is sensitive")
    description: Optional[str] = Field(None, description="Configuration description")


class AdminConfigResponse(BaseResponse):
    """Admin configuration response model."""
    status: str = Field(default="success", description="Success status")
    configuration: List[SystemConfigItem] = Field(..., description="System configuration items")
    environment: str = Field(..., description="Current environment")
    config_loaded_at: datetime = Field(default_factory=datetime.utcnow, description="Configuration load time")


class LogEntry(BaseModel):
    """Log entry model."""
    timestamp: datetime = Field(..., description="Log entry timestamp")
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    component: Optional[str] = Field(None, description="Component that generated the log")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional log context")


class AdminLogsResponse(BaseResponse):
    """Admin logs response model."""
    status: str = Field(default="success", description="Success status")
    logs: List[LogEntry] = Field(..., description="Recent log entries")
    total_logs: int = Field(..., ge=0, description="Total number of logs")
    log_level_filter: Optional[str] = Field(None, description="Applied log level filter")
    time_range: str = Field(..., description="Time range for logs")


# Export all models
__all__ = [
    # Enums
    "DocumentStatus", "EscalationReason", "FeedbackCategory",
    # Base models
    "BaseResponse", "ErrorDetail", "ErrorResponse",
    # Document models
    "DocumentMetadata", "DocumentUploadRequest", "DocumentContentRequest", "DocumentChunk",
    "DocumentResponse", "DocumentListItem", "DocumentListResponse", "DocumentDeleteResponse",
    # Chat/RAG models
    "ChatRequest", "SourceDocument", "ChatResponse",
    # Search models
    "SearchRequest", "SearchResult", "SearchResponse",
    # Escalation models
    "EscalationRequest", "EscalationResponse",
    # Feedback models
    "FeedbackRequest", "FeedbackResponse",
    # Admin models
    "SystemStats", "VectorStoreStats", "AdminStatsResponse",
    "SystemHealthDetail", "AdminHealthResponse",
    "SystemConfigItem", "AdminConfigResponse",
    "LogEntry", "AdminLogsResponse"
]
