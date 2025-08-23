"""
SAIA-RAG FastAPI Application

Main application entry point with health checks, error handling, and logging.
Follows clean architecture patterns with proper dependency injection.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
import structlog

from .config import get_settings, Settings
from .vector_store import get_vector_store
from .models import (
    DocumentUploadRequest, DocumentResponse, DocumentListResponse,
    DocumentDeleteResponse, ErrorResponse, ChatRequest, ChatResponse,
    SearchRequest, SearchResponse, EscalationRequest, EscalationResponse,
    FeedbackRequest, FeedbackResponse
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger()


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    # Configure app based on environment
    app_config = {
        "title": settings.app_name,
        "description": "RAG-based customer support chatbot with smart fallbacks",
        "version": settings.app_version,
    }

    # Disable docs in production
    if settings.is_production():
        app_config.update({
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None
        })

    app = FastAPI(**app_config)

    logger.info(
        "FastAPI application created",
        environment=settings.environment,
        debug=settings.debug,
        version=settings.app_version
    )

    return app


# Create application instance
app = create_application()


# === APPLICATION EVENTS ===

@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    settings = get_settings()
    logger.info(
        "SAIA-RAG application starting up",
        environment=settings.environment,
        debug=settings.debug,
        tenant_id=settings.tenant_id,
        version=settings.app_version
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    logger.info("SAIA-RAG application shutting down")


# === EXCEPTION HANDLERS ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url.path),
        method=request.method
    )
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url.path)
        }
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions with proper logging."""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        path=str(request.url.path),
        method=request.method,
        exc_info=True
    )
    return {
        "error": {
            "code": 500,
            "message": "Internal server error",
            "path": str(request.url.path)
        }
    }


# === HEALTH CHECK ENDPOINTS ===

@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """
    Health check endpoint for service monitoring.

    Returns comprehensive health status including dependencies.

    Returns:
        Health status with service information and dependency checks
    """
    try:
        # Check vector store health
        vector_store = get_vector_store()
        vector_health = await vector_store.health_check()

        # Determine overall status based on dependencies
        overall_status = "ok" if vector_health["status"] == "healthy" else "degraded"

        # Basic service health
        health_data = {
            "status": overall_status,
            "service": "SAIA-RAG API",
            "version": settings.app_version,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.environment,
            "dependencies": {
                "vector_store": vector_health
            }
        }

        logger.info("Health check requested", status="ok")
        return health_data

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Service health check failed"
        )


@app.get("/")
async def root(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """
    Root endpoint with service information.

    Returns:
        Basic service information and available endpoints
    """
    try:
        response_data = {
            "message": settings.app_name,
            "status": "running",
            "version": settings.app_version,
            "environment": settings.environment,
        }

        # Add docs URL only in development
        if settings.is_development():
            response_data["docs"] = "/docs"
            response_data["redoc"] = "/redoc"

        logger.info("Root endpoint accessed")
        return response_data

    except Exception as e:
        logger.error("Root endpoint failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# === DOCUMENT MANAGEMENT ENDPOINTS ===

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    request: DocumentUploadRequest,
    settings: Settings = Depends(get_settings)
) -> DocumentResponse:
    """
    Upload and process a document for the RAG knowledge base.

    This endpoint accepts document metadata and returns processing information.
    The actual file upload and processing would be implemented with multipart/form-data.

    Args:
        request: Document upload request with metadata
        settings: Application settings

    Returns:
        Document processing result with chunk information

    Raises:
        HTTPException: If document processing fails
    """
    try:
        start_time = datetime.utcnow()

        # Generate unique document ID
        import uuid
        document_id = str(uuid.uuid4())

        # TODO: Implement actual file processing
        # - Extract text from uploaded file
        # - Split into chunks
        # - Generate embeddings via OpenAI
        # - Index in vector store

        # For now, simulate processing
        chunks_created = 5  # Simulated chunk count

        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Document upload processed",
            document_id=document_id,
            title=request.metadata.title,
            category=request.metadata.category,
            chunks_created=chunks_created,
            processing_time_ms=processing_time_ms
        )

        return DocumentResponse(
            document_id=document_id,
            title=request.metadata.title,
            category=request.metadata.category,
            chunks_created=chunks_created,
            processing_time_ms=processing_time_ms,
            upload_date=start_time
        )

    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Document upload processing failed"
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
    search: Optional[str] = None,
    settings: Settings = Depends(get_settings)
) -> DocumentListResponse:
    """
    List uploaded documents with optional filtering.

    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip
        category: Filter by document category
        search: Search documents by title or content
        settings: Application settings

    Returns:
        List of documents with metadata and pagination info

    Raises:
        HTTPException: If document listing fails
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
        if offset < 0:
            raise HTTPException(status_code=400, detail="Offset must be non-negative")

        # TODO: Implement actual document listing
        # - Query vector store for document metadata
        # - Apply filters (category, search)
        # - Implement pagination
        # - Return document list with metadata

        # For now, return empty list
        documents = []
        total = 0

        logger.info(
            "Documents listed",
            limit=limit,
            offset=offset,
            category=category,
            search=search,
            total_found=total
        )

        return DocumentListResponse(
            documents=documents,
            total=total,
            limit=limit,
            offset=offset
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document listing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Document listing failed"
        )


@app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    settings: Settings = Depends(get_settings)
) -> DocumentDeleteResponse:
    """
    Delete a document and all its associated chunks.

    Args:
        document_id: Unique document identifier to delete
        settings: Application settings

    Returns:
        Deletion confirmation with chunk count

    Raises:
        HTTPException: If document deletion fails or document not found
    """
    try:
        # Get vector store instance
        vector_store = get_vector_store()

        # Check if document exists by trying to get its chunks
        try:
            chunks = vector_store.get_document_chunks(document_id)
            if not chunks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {document_id} not found"
                )
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )

        # Delete the document
        delete_result = vector_store.delete_document(document_id)
        chunks_deleted = len(chunks)

        logger.info(
            "Document deleted successfully",
            document_id=document_id,
            chunks_deleted=chunks_deleted
        )

        return DocumentDeleteResponse(
            document_id=document_id,
            chunks_deleted=chunks_deleted
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Document deletion failed",
            document_id=document_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Document deletion failed"
        )


# === RAG QUERY ENDPOINTS ===

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings)
) -> ChatResponse:
    """
    Process a chat message using RAG (Retrieval-Augmented Generation).

    This endpoint:
    1. Converts the user message to embeddings
    2. Searches for relevant document chunks
    3. Generates a response using OpenAI with context
    4. Returns the response with source citations

    Args:
        request: Chat request with message and options
        settings: Application settings

    Returns:
        AI-generated response with sources and metadata

    Raises:
        HTTPException: If chat processing fails
    """
    try:
        start_time = datetime.utcnow()

        # TODO: Implement actual RAG pipeline
        # 1. Generate embeddings for user message via OpenAI
        # 2. Search vector store for relevant chunks
        # 3. Build context from retrieved chunks
        # 4. Generate response via OpenAI Chat API
        # 5. Extract and format source citations

        # For now, simulate RAG response
        response_text = f"I understand you're asking about: '{request.message}'. This is a simulated response from the RAG system."
        confidence = 0.85
        sources = []  # Would contain actual source documents

        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Chat request processed",
            message_length=len(request.message),
            conversation_id=request.conversation_id,
            confidence=confidence,
            sources_count=len(sources),
            processing_time_ms=processing_time_ms
        )

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            confidence=confidence,
            sources=sources,
            processing_time_ms=processing_time_ms,
            tokens_used=150  # Simulated token count
        )

    except Exception as e:
        logger.error(
            "Chat request failed",
            message=request.message[:100],
            conversation_id=request.conversation_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Chat request processing failed"
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    settings: Settings = Depends(get_settings)
) -> SearchResponse:
    """
    Search documents using vector similarity.

    This endpoint:
    1. Converts the search query to embeddings
    2. Performs vector similarity search in Qdrant
    3. Returns ranked results with relevance scores

    Args:
        request: Search request with query and filters
        settings: Application settings

    Returns:
        Search results with relevance scores and metadata

    Raises:
        HTTPException: If search processing fails
    """
    try:
        start_time = datetime.utcnow()

        # Get vector store instance
        vector_store = get_vector_store()

        # TODO: Implement actual search pipeline
        # 1. Generate embeddings for search query via OpenAI
        # 2. Perform vector search in Qdrant
        # 3. Apply filters if provided
        # 4. Format and return results

        # For now, simulate search results
        results = []
        total_results = 0

        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Search request processed",
            query=request.query,
            limit=request.limit,
            min_score=request.min_score,
            results_count=len(results),
            processing_time_ms=processing_time_ms
        )

        return SearchResponse(
            results=results,
            total_results=total_results,
            processing_time_ms=processing_time_ms,
            query=request.query
        )

    except Exception as e:
        logger.error(
            "Search request failed",
            query=request.query,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Search request processing failed"
        )


# === ESCALATION ENDPOINT ===

@app.post("/escalate", response_model=EscalationResponse)
async def escalate(
    request: EscalationRequest,
    settings: Settings = Depends(get_settings)
) -> EscalationResponse:
    """
    Escalate a conversation to human support.

    This endpoint creates a support ticket and notifies the support team
    when the AI assistant cannot adequately help the user.

    Args:
        request: Escalation request with reason and context
        settings: Application settings

    Returns:
        Escalation confirmation with ticket information

    Raises:
        HTTPException: If escalation processing fails
    """
    try:
        # Generate unique escalation ID
        import uuid
        escalation_id = str(uuid.uuid4())

        # Generate ticket number (would integrate with ticketing system)
        ticket_number = f"SAIA-{escalation_id[:8].upper()}"

        # TODO: Implement actual escalation logic
        # - Create ticket in support system (Jira, Zendesk, etc.)
        # - Send notification to support team
        # - Store escalation context and conversation history
        # - Set up follow-up tracking

        # Determine estimated response time based on priority
        response_times = {
            "urgent": "1-2 hours",
            "high": "4-6 hours",
            "medium": "1-2 business days",
            "low": "3-5 business days"
        }
        estimated_response = response_times.get(request.priority, "1-2 business days")

        logger.info(
            "Escalation created",
            escalation_id=escalation_id,
            ticket_number=ticket_number,
            reason=request.reason.value,
            priority=request.priority,
            conversation_id=request.conversation_id
        )

        return EscalationResponse(
            escalation_id=escalation_id,
            ticket_number=ticket_number,
            estimated_response_time=estimated_response
        )

    except Exception as e:
        logger.error(
            "Escalation failed",
            reason=request.reason.value if request.reason else None,
            priority=request.priority,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Escalation processing failed"
        )


# === FEEDBACK ENDPOINT ===

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest,
    settings: Settings = Depends(get_settings)
) -> FeedbackResponse:
    """
    Submit feedback on AI assistant responses.

    This endpoint collects user feedback to improve the system's
    performance and response quality over time.

    Args:
        request: Feedback request with rating and comments
        settings: Application settings

    Returns:
        Feedback confirmation

    Raises:
        HTTPException: If feedback processing fails
    """
    try:
        # Generate unique feedback ID
        import uuid
        feedback_id = str(uuid.uuid4())

        # TODO: Implement actual feedback processing
        # - Store feedback in database
        # - Analyze feedback patterns
        # - Update model performance metrics
        # - Trigger retraining if needed
        # - Send feedback to analytics system

        # Generate appropriate response message based on rating
        if request.rating >= 4:
            message = "Thank you for your positive feedback! We're glad we could help."
        elif request.rating >= 3:
            message = "Thank you for your feedback. We'll continue working to improve our responses."
        else:
            message = "Thank you for your feedback. We take your concerns seriously and will work to improve."

        logger.info(
            "Feedback received",
            feedback_id=feedback_id,
            rating=request.rating,
            category=request.category.value,
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            has_text_feedback=bool(request.feedback)
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            message=message
        )

    except Exception as e:
        logger.error(
            "Feedback processing failed",
            rating=request.rating if hasattr(request, 'rating') else None,
            category=request.category.value if hasattr(request, 'category') and request.category else None,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Feedback processing failed"
        )
