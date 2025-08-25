"""
SAIA-RAG FastAPI Application

Main application entry point with health checks, error handling, and logging.
Follows clean architecture patterns with proper dependency injection.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import time
import asyncio

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import structlog

from .config import get_settings, Settings
from .vector_store import get_vector_store
from .rag_service import get_rag_service
from .openai_client import get_openai_client
from .whatsapp_client import get_whatsapp_client
from .middleware import setup_middleware
from .models import (
DocumentUploadRequest, DocumentContentRequest, DocumentResponse, DocumentListResponse,
DocumentDeleteResponse, ErrorResponse, ChatRequest, ChatResponse, SourceDocument,
SearchRequest, SearchResponse, EscalationRequest, EscalationResponse,
FeedbackRequest, FeedbackResponse, AdminStatsResponse, AdminHealthResponse,
AdminConfigResponse, AdminLogsResponse, SystemStats, VectorStoreStats,
SystemHealthDetail, SystemConfigItem, LogEntry
)

# Configure structured logging with proper keyword argument handling
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
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

    # Mount static files for web UI
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Setup middleware in correct order
    setup_middleware(app)

    logger.info(
        "FastAPI application created",
        environment=settings.environment,
        debug=settings.debug,
        version=settings.app_version
    )

    return app


# Create application instance
app = create_application()


# === AUTHENTICATION ===

async def api_key_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """
    API key authentication dependency.

    Args:
        request: FastAPI request object
        authorization: Authorization header value
        settings: Application settings

    Returns:
        API key if valid, None if not required (dev only)

    Raises:
        HTTPException: If authentication fails in production
    """
    # Skip authentication in development mode only
    if settings.is_development():
        return None

    # Production mode - API key required for all protected endpoints
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Extract API key from Bearer token
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    # Validate API key
    if not settings.api_key or token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token


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
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path)
            }
        }
    )


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
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "path": str(request.url.path)
            }
        }
    )


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

        # Check OpenAI health
        openai_client = get_openai_client()
        openai_health = await openai_client.health_check()

        # Determine overall status based on dependencies
        all_healthy = (
            vector_health["status"] == "healthy" and
            openai_health["status"] == "healthy"
        )
        overall_status = "ok" if all_healthy else "degraded"

        # Basic service health
        health_data = {
            "status": overall_status,
            "service": "SAIA-RAG API",
            "version": settings.app_version,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.environment,
            "dependencies": {
                "vector_store": vector_health,
                "openai": openai_health
            }
        }

        logger.info("Health check requested", status="ok")
        return JSONResponse(content=health_data)

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
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error("Root endpoint failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """
    Serve the web UI for SAIA-RAG chat interface.

    Returns:
        HTML response with the chat interface
    """
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Web UI not found"
        )
    except Exception as e:
        logger.error("Failed to serve web UI", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to load web UI"
        )






# === DOCUMENT MANAGEMENT ENDPOINTS ===

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    request: DocumentUploadRequest,
    content: Optional[str] = None,
    settings: Settings = Depends(get_settings)
) -> DocumentResponse:
    """
    Upload and process a document for the RAG knowledge base.

    This endpoint processes document content through the complete RAG pipeline:
    - Text extraction and cleaning
    - Intelligent chunking with overlap
    - OpenAI embedding generation
    - Vector store indexing

    Args:
        request: Document upload request with metadata
        content: Document content (optional - uses sample content if not provided)
        settings: Application settings

    Returns:
        Document processing result with chunk information

    Raises:
        HTTPException: If document processing fails
    """
    try:
        start_time = datetime.utcnow()

        # Use provided content or default sample content
        document_content = content or """
    Password Reset Instructions:

    1. Go to the login page and click "Forgot Password"
    2. Enter your email address
    3. Check your email for a reset link
    4. Click the link and create a new password
    5. Your password must be at least 8 characters long
    6. Include uppercase, lowercase, numbers, and special characters

    If you continue to have issues:
    - Clear your browser cache and cookies
    - Try using a different browser
    - Contact support if the problem persists

    Account Management:
    - You can update your profile information in Account Settings
    - Enable two-factor authentication for better security
    - Review your login history regularly

    Troubleshooting Common Issues:
    - "Invalid credentials" error: Check caps lock and typing
    - "Account locked" message: Wait 15 minutes or contact support
    - Email not received: Check spam folder and verify email address
    """

        # Get RAG service
        rag_service = get_rag_service()

        # Process document through RAG pipeline
        ingestion_result = await rag_service.ingest_document(
            content=document_content,
            metadata=request.metadata.dict(),
            content_type="text/plain"
        )

        if not ingestion_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Document processing failed: {ingestion_result['error']}"
            )

        end_time = datetime.utcnow()
        total_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Document upload completed",
            document_id=ingestion_result["document_id"],
            title=request.metadata.title,
            category=request.metadata.category,
            chunks_created=ingestion_result["chunks_created"],
            total_time_ms=total_time_ms
        )

        return DocumentResponse(
            document_id=ingestion_result["document_id"],
            title=request.metadata.title,
            category=request.metadata.category,
            chunks_created=ingestion_result["chunks_created"],
            processing_time_ms=total_time_ms,
            upload_date=start_time
        )

    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Document upload failed"
        )


@app.post("/documents/upload-content", response_model=DocumentResponse)
async def upload_document_with_content(
    request: DocumentContentRequest,
    settings: Settings = Depends(get_settings)
) -> DocumentResponse:
    """
    Upload a document with custom content.

    This endpoint allows you to upload documents with your own content
    instead of using the default sample content.

    Args:
        request: Document content upload request
        settings: Application settings

    Returns:
        Document processing result with chunk information

    Raises:
        HTTPException: If document processing fails
    """
    try:
        start_time = datetime.utcnow()

        # Create metadata
        metadata = {
            "title": request.title,
            "category": request.category,
            "tags": request.tags,
            "author": request.author
        }

        # Get RAG service
        rag_service = get_rag_service()

        # Process document through RAG pipeline
        ingestion_result = await rag_service.ingest_document(
            content=request.content,
            metadata=metadata,
            content_type="text/plain"
        )

        if not ingestion_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Document processing failed: {ingestion_result['error']}"
            )

        end_time = datetime.utcnow()
        total_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Document with content uploaded",
            document_id=ingestion_result["document_id"],
            title=request.title,
            category=request.category,
            content_length=len(request.content),
            chunks_created=ingestion_result["chunks_created"],
            total_time_ms=total_time_ms
        )

        return DocumentResponse(
            document_id=ingestion_result["document_id"],
            title=request.title,
            category=request.category,
            chunks_created=ingestion_result["chunks_created"],
            processing_time_ms=total_time_ms,
            upload_date=start_time
        )

    except Exception as e:
        logger.error("Document content upload failed", error=str(e))
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

        # Get vector store instance
        vector_store = get_vector_store()

        # Get all documents from vector store
        all_documents = vector_store.list_documents()

        # Apply category filter if provided
        if category:
            all_documents = [doc for doc in all_documents if doc.get('category') == category]

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            all_documents = [
                doc for doc in all_documents
                if search_lower in (doc.get('title', '') or '').lower() or
                   search_lower in (doc.get('category', '') or '').lower() or
                   any(search_lower in tag.lower() for tag in doc.get('tags', []))
            ]

        # Calculate pagination
        total = len(all_documents)
        start_idx = offset
        end_idx = min(offset + limit, total)
        paginated_documents = all_documents[start_idx:end_idx]

        # Convert to DocumentListItem format
        from .models import DocumentListItem, DocumentStatus
        documents = []
        for doc in paginated_documents:
            documents.append(DocumentListItem(
                document_id=doc['document_id'],
                title=doc.get('title'),
                category=doc.get('category'),
                tags=doc.get('tags', []),
                chunk_count=doc.get('chunk_count', 0),
                status=DocumentStatus.PROCESSED,
                upload_date=datetime.fromisoformat(doc['upload_date']) if doc.get('upload_date') else datetime.utcnow(),
                author=doc.get('author')
            ))

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
    api_key: Optional[str] = Depends(api_key_auth),
    settings: Settings = Depends(get_settings)
) -> ChatResponse:
    """
    Process a chat message using RAG (Retrieval-Augmented Generation).

    This endpoint implements the complete RAG pipeline:
    1. Converts the user message to embeddings via OpenAI
    2. Searches vector store for relevant document chunks
    3. Builds context from retrieved chunks
    4. Generates response using OpenAI Chat API with context
    5. Returns response with source citations

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
        logger.info("Chat request started", message_preview=request.message[:50])

        # Get RAG service
        rag_service = get_rag_service()
        logger.info("RAG service obtained")

        # Generate RAG response
        logger.info("Starting RAG response generation")
        rag_result = await rag_service.generate_response(
            query=request.message,
            conversation_id=request.conversation_id,
            max_context_chunks=8,
            confidence_threshold=settings.confidence_threshold,
            channel="chat"
        )
        logger.info("RAG response generated", result_keys=list(rag_result.keys()))

        end_time = datetime.utcnow()
        total_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Chat request processed via RAG",
            message_length=len(request.message),
            conversation_id=request.conversation_id,
            confidence=rag_result["confidence"],
            sources_count=rag_result["sources_count"],
            processing_time_ms=total_time_ms,
            tokens_used=rag_result["tokens_used"]
        )

        # Debug the sources before creating response
        logger.info("Sources debug",
                   sources_type=type(rag_result["sources"]),
                   sources_length=len(rag_result["sources"]) if rag_result["sources"] else 0)

        if rag_result["sources"]:
            logger.info("First source debug",
                       first_source_type=type(rag_result["sources"][0]),
                       first_source_keys=list(rag_result["sources"][0].__dict__.keys()) if hasattr(rag_result["sources"][0], '__dict__') else "no __dict__")

        logger.info("Creating ChatResponse object")
        chat_response = ChatResponse(
            response=rag_result["response"],
            conversation_id=request.conversation_id,
            confidence=rag_result["confidence"],
            sources=rag_result["sources"],
            processing_time_ms=total_time_ms,
            tokens_used=rag_result["tokens_used"]
        )
        logger.info("ChatResponse created successfully")

        return chat_response

    except Exception as e:
        logger.error(
            "Chat request failed",
            message=request.message[:100],
            conversation_id=request.conversation_id,
            error=str(e),
            error_type=type(e).__name__,
            traceback=str(e.__traceback__)
        )

        # Return a proper error response instead of raising HTTPException
        return ChatResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again later.",
            conversation_id=request.conversation_id,
            confidence=0.0,
            sources=[],
            processing_time_ms=0,
            tokens_used=0
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    settings: Settings = Depends(get_settings)
) -> SearchResponse:
    """
    Search documents using vector similarity.

    This endpoint implements semantic search:
    1. Converts the search query to embeddings via OpenAI
    2. Performs vector similarity search in Qdrant
    3. Applies filters if provided
    4. Returns ranked results with relevance scores

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

        # Get RAG service
        rag_service = get_rag_service()

        # Perform semantic search
        search_result = await rag_service.search_documents(
            query=request.query,
            limit=request.limit,
            min_score=request.min_score,
            filters=request.filters
        )

        end_time = datetime.utcnow()
        total_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Search request processed via RAG",
            query=request.query,
            limit=request.limit,
            min_score=request.min_score,
            results_count=search_result["total_results"],
            processing_time_ms=total_time_ms
        )

        return SearchResponse(
            results=search_result["results"],
            total_results=search_result["total_results"],
            processing_time_ms=total_time_ms,
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
    api_key: Optional[str] = Depends(api_key_auth),
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
    api_key: Optional[str] = Depends(api_key_auth),
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


# === ADMIN ENDPOINTS ===

@app.get("/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    settings: Settings = Depends(get_settings)
) -> AdminStatsResponse:
    """
    Get comprehensive system statistics for administrators.

    This endpoint provides detailed statistics about:
    - Document and chunk counts
    - Conversation and escalation metrics
    - System performance metrics
    - Vector store statistics

    Args:
        settings: Application settings

    Returns:
        Comprehensive system statistics

    Raises:
        HTTPException: If statistics collection fails
    """
    try:
        start_time = datetime.utcnow()

        # Get vector store instance
        vector_store = get_vector_store()

        # TODO: Implement actual statistics collection
        # - Query database for document/conversation counts
        # - Calculate performance metrics
        # - Collect system resource usage
        # - Aggregate escalation and feedback statistics

        # For now, simulate statistics
        system_stats = SystemStats(
            total_documents=0,  # Would query actual document count
            total_chunks=0,     # Would query vector store point count
            total_conversations=0,  # Would query conversation history
            total_escalations=0,    # Would query escalation records
            total_feedback=0,       # Would query feedback records
            avg_response_time_ms=150.5,  # Would calculate from metrics
            uptime_seconds=3600  # Would calculate actual uptime
        )

        # Get vector store statistics
        try:
            collection_info = vector_store.get_collection_info()
            vector_stats = VectorStoreStats(
                collection_name=collection_info["collection_name"],
                total_points=collection_info["points_count"],
                vector_size=collection_info["config"]["vector_size"],
                distance_metric=collection_info["config"]["distance"],
                index_status="ready",
                memory_usage_mb=None  # Would get from Qdrant metrics
            )
        except Exception as e:
            logger.warning("Failed to get vector store stats", error=str(e))
            vector_stats = VectorStoreStats(
                collection_name=settings.get_collection_name(),
                total_points=0,
                vector_size=settings.embed_dim,
                distance_metric="COSINE",
                index_status="unknown"
            )

        end_time = datetime.utcnow()
        collection_time_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "Admin statistics collected",
            collection_time_ms=collection_time_ms,
            total_documents=system_stats.total_documents,
            total_chunks=vector_stats.total_points
        )

        return AdminStatsResponse(
            system=system_stats,
            vector_store=vector_stats,
            collection_time=start_time
        )

    except Exception as e:
        logger.error("Admin statistics collection failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Statistics collection failed"
        )


@app.get("/admin/health", response_model=AdminHealthResponse)
async def get_admin_health(
    settings: Settings = Depends(get_settings)
) -> AdminHealthResponse:
    """
    Get detailed system health information for administrators.

    This endpoint provides comprehensive health checks for:
    - Vector store connectivity and performance
    - API response times
    - System resource usage
    - External service dependencies

    Args:
        settings: Application settings

    Returns:
        Detailed system health information

    Raises:
        HTTPException: If health check fails
    """
    try:
        components = []
        overall_healthy = True

        # Check vector store health
        vector_store = get_vector_store()
        vector_health = await vector_store.health_check()

        vector_component = SystemHealthDetail(
            component="vector_store",
            status=vector_health["status"],
            response_time_ms=vector_health.get("response_time_ms"),
            details={
                "url": vector_health.get("url"),
                "collection_name": vector_health.get("collection_name"),
                "collections_count": vector_health.get("collections_count")
            }
        )
        components.append(vector_component)

        if vector_health["status"] != "healthy":
            overall_healthy = False

        # TODO: Add more component health checks
        # - Database connectivity
        # - External API dependencies (OpenAI)
        # - File system access
        # - Memory and CPU usage
        # - Network connectivity

        # API health check (self)
        api_component = SystemHealthDetail(
            component="api_server",
            status="healthy",
            response_time_ms=1,  # Self-check is always fast
            details={
                "version": settings.app_version,
                "environment": settings.environment,
                "debug_mode": settings.debug
            }
        )
        components.append(api_component)

        overall_status = "healthy" if overall_healthy else "degraded"

        logger.info(
            "Admin health check completed",
            overall_status=overall_status,
            components_checked=len(components)
        )

        return AdminHealthResponse(
            overall_health=overall_status,
            components=components,
            checks_performed=len(components)
        )

    except Exception as e:
        logger.error("Admin health check failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@app.get("/admin/config", response_model=AdminConfigResponse)
async def get_admin_config(
    settings: Settings = Depends(get_settings)
) -> AdminConfigResponse:
    """
    Get system configuration for administrators.

    This endpoint provides sanitized system configuration information
    for debugging and monitoring purposes. Sensitive values are masked.

    Args:
        settings: Application settings

    Returns:
        System configuration information

    Raises:
        HTTPException: If configuration retrieval fails
    """
    try:
        config_items = []

        # Application settings
        config_items.extend([
            SystemConfigItem(
                key="app_version",
                value=settings.app_version,
                category="application",
                description="Application version"
            ),
            SystemConfigItem(
                key="environment",
                value=settings.environment,
                category="application",
                description="Runtime environment"
            ),
            SystemConfigItem(
                key="debug",
                value=str(settings.debug),
                category="application",
                description="Debug mode enabled"
            ),
            SystemConfigItem(
                key="tenant_id",
                value=settings.tenant_id,
                category="application",
                description="Tenant identifier"
            )
        ])

        # OpenAI settings (sanitized)
        config_items.extend([
            SystemConfigItem(
                key="openai_api_key",
                value="sk-***" if settings.openai_api_key else "not_set",
                category="openai",
                is_sensitive=True,
                description="OpenAI API key status"
            ),
            SystemConfigItem(
                key="openai_chat_model",
                value=settings.openai_chat_model,
                category="openai",
                description="OpenAI chat model"
            ),
            SystemConfigItem(
                key="openai_embed_model",
                value=settings.openai_embed_model,
                category="openai",
                description="OpenAI embedding model"
            ),
            SystemConfigItem(
                key="embed_dim",
                value=str(settings.embed_dim),
                category="openai",
                description="Embedding dimensions"
            )
        ])

        # Qdrant settings
        config_items.extend([
            SystemConfigItem(
                key="qdrant_url",
                value=settings.qdrant_url,
                category="qdrant",
                description="Qdrant server URL"
            ),
            SystemConfigItem(
                key="collection_name",
                value=settings.get_collection_name(),
                category="qdrant",
                description="Vector collection name"
            ),
            SystemConfigItem(
                key="confidence_threshold",
                value=str(settings.confidence_threshold),
                category="qdrant",
                description="Confidence threshold for responses"
            )
        ])

        logger.info(
            "Admin configuration retrieved",
            config_items_count=len(config_items),
            environment=settings.environment
        )

        return AdminConfigResponse(
            configuration=config_items,
            environment=settings.environment
        )

    except Exception as e:
        logger.error("Admin configuration retrieval failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Configuration retrieval failed"
        )


@app.get("/admin/logs", response_model=AdminLogsResponse)
async def get_admin_logs(
    limit: int = 100,
    level: Optional[str] = None,
    component: Optional[str] = None,
    settings: Settings = Depends(get_settings)
) -> AdminLogsResponse:
    """
    Get recent system logs for administrators.

    This endpoint provides access to recent system logs for debugging
    and monitoring purposes.

    Args:
        limit: Maximum number of log entries to return (1-1000)
        level: Filter by log level (debug, info, warning, error)
        component: Filter by component name
        settings: Application settings

    Returns:
        Recent system log entries

    Raises:
        HTTPException: If log retrieval fails
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")

        if level and level.lower() not in ["debug", "info", "warning", "error"]:
            raise HTTPException(status_code=400, detail="Invalid log level")

        # TODO: Implement actual log retrieval
        # - Read from log files or log aggregation system
        # - Apply filters (level, component, time range)
        # - Parse and format log entries
        # - Return structured log data

        # For now, simulate recent logs
        logs = [
            LogEntry(
                timestamp=datetime.utcnow(),
                level="info",
                message="Admin logs endpoint accessed",
                component="api_server",
                context={"limit": limit, "level": level, "component": component}
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level="info",
                message="Vector store health check completed",
                component="vector_store",
                context={"status": "healthy", "response_time_ms": 3}
            )
        ]

        # Apply filters if provided
        if level:
            logs = [log for log in logs if log.level.lower() == level.lower()]

        if component:
            logs = [log for log in logs if log.component == component]

        # Limit results
        logs = logs[:limit]

        time_range = "last_24_hours"  # Would be configurable

        logger.info(
            "Admin logs retrieved",
            logs_returned=len(logs),
            limit=limit,
            level_filter=level,
            component_filter=component
        )

        return AdminLogsResponse(
            logs=logs,
            total_logs=len(logs),
            log_level_filter=level,
            time_range=time_range
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Admin logs retrieval failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Logs retrieval failed"
        )


@app.get("/admin/metrics")
async def get_system_metrics(
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Get comprehensive system performance metrics.

    This endpoint provides detailed performance and usage metrics
    for system monitoring and optimization.

    Args:
        settings: Application settings

    Returns:
        System performance metrics and statistics
    """
    try:
        start_time = datetime.utcnow()

        # Get RAG service for cache metrics
        rag_service = get_rag_service()

        # System metrics
        metrics = {
            "timestamp": start_time.isoformat(),
            "system": {
                "uptime_seconds": int((start_time - datetime.utcnow()).total_seconds()),
                "environment": settings.environment,
                "version": settings.app_version,
                "debug_mode": settings.debug
            },
            "cache": {
                "query_cache_size": len(rag_service._query_cache),
                "cache_max_size": rag_service._cache_max_size,
                "cache_hit_ratio": "N/A"  # Would need to track hits/misses
            },
            "configuration": {
                "openai_chat_model": settings.openai_chat_model,
                "openai_embed_model": settings.openai_embed_model,
                "embed_dim": settings.embed_dim,
                "confidence_threshold": settings.confidence_threshold,
                "tenant_id": settings.tenant_id
            },
            "endpoints": {
                "total_endpoints": 13,  # Current endpoint count
                "health_check": "/health",
                "chat": "/chat",
                "documents": "/documents",
                "search": "/search",
                "admin": "/admin/*"
            }
        }

        # Get vector store metrics
        try:
            vector_store = get_vector_store()
            collection_info = vector_store.get_collection_info()
            metrics["vector_store"] = {
                "collection_name": collection_info["collection_name"],
                "total_points": collection_info["points_count"],
                "vector_size": collection_info["config"]["vector_size"],
                "distance_metric": collection_info["config"]["distance"]
            }
        except Exception as e:
            metrics["vector_store"] = {
                "status": "error",
                "error": str(e)
            }

        # Get OpenAI client metrics
        try:
            openai_client = get_openai_client()
            openai_health = await openai_client.health_check()
            metrics["openai"] = {
                "status": openai_health["status"],
                "chat_model": openai_health["chat_model"],
                "embed_model": openai_health["embed_model"],
                "embed_dim": openai_health["embed_dim"]
            }
        except Exception as e:
            metrics["openai"] = {
                "status": "error",
                "error": str(e)
            }

        end_time = datetime.utcnow()
        collection_time_ms = int((end_time - start_time).total_seconds() * 1000)
        metrics["collection_time_ms"] = collection_time_ms

        logger.info(
            "System metrics collected",
            collection_time_ms=collection_time_ms,
            cache_size=metrics["cache"]["query_cache_size"]
        )

        return JSONResponse(content=metrics)

    except Exception as e:
        logger.error("System metrics collection failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Metrics collection failed"
        )


# === WHATSAPP BUSINESS API ENDPOINTS ===

@app.get("/whatsapp/webhook")
async def whatsapp_webhook_verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    settings: Settings = Depends(get_settings)
):
    """
    WhatsApp webhook verification endpoint.

    This endpoint is called by WhatsApp to verify the webhook URL
    during the initial setup process.

    Args:
        hub_mode: Verification mode from WhatsApp
        hub_verify_token: Verification token from WhatsApp
        hub_challenge: Challenge string from WhatsApp
        settings: Application settings

    Returns:
        Challenge string if verification succeeds

    Raises:
        HTTPException: If verification fails
    """
    try:
        if not settings.is_whatsapp_configured():
            raise HTTPException(
                status_code=503,
                detail="WhatsApp integration not configured"
            )

        whatsapp_client = get_whatsapp_client()
        challenge = whatsapp_client.verify_webhook(
            mode=hub_mode,
            token=hub_verify_token,
            challenge=hub_challenge
        )

        if challenge:
            logger.info("WhatsApp webhook verification successful")
            # Return plain text response (not JSON) as required by Meta
            return PlainTextResponse(content=challenge)
        else:
            logger.warning("WhatsApp webhook verification failed")
            raise HTTPException(
                status_code=403,
                detail="Webhook verification failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("WhatsApp webhook verification error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Webhook verification failed"
        )


@app.post("/whatsapp/webhook")
async def whatsapp_webhook_receive(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    """
    WhatsApp webhook message receiver endpoint.

    Following Meta's best practices:
    - Responds with 200 OK within 80ms
    - Processes business logic asynchronously
    - Uses background tasks for RAG processing

    Args:
        request: FastAPI request object
        settings: Application settings

    Returns:
        Immediate 200 OK response
    """
    try:
        # Step 1: Respond immediately (within 80ms as per Meta requirements)
        if not settings.is_whatsapp_configured():
            logger.warning("WhatsApp webhook received but not configured")
            return JSONResponse(content={"status": "not_configured"})

        # Parse request body quickly
        request_data = await request.json()

        logger.info("WhatsApp webhook received",
                   entry_count=len(request_data.get("entry", [])))

        # Step 2: Quick validation and immediate response
        whatsapp_client = get_whatsapp_client()
        message_data = whatsapp_client.parse_webhook_message(request_data)

        logger.info("WhatsApp message parsing result",
                   message_data_exists=message_data is not None,
                   message_data_keys=list(message_data.keys()) if message_data else None)

        if not message_data:
            # Not a text message, status update, or parsing failed
            logger.info("WhatsApp message ignored - no valid message data")
            return JSONResponse(content={"status": "ignored"})

        # Step 3: Schedule async processing (don't wait for it)
        logger.info("Scheduling WhatsApp background task",
                   from_number=message_data.get("from"),
                   message_preview=message_data.get("text", "")[:50])

        background_tasks.add_task(
            process_whatsapp_message_async,
            message_data=message_data,
            settings=settings
        )

        logger.info("WhatsApp background task scheduled successfully")

        # Step 4: Return immediate success (as required by Meta)
        return JSONResponse(content={"status": "received"})

    except Exception as e:
        # Log error but still return 200 to prevent webhook retries
        logger.error("WhatsApp webhook processing error", error=str(e))
        return JSONResponse(content={"status": "error", "message": "Webhook processed with errors"})


async def process_whatsapp_message_async(
    message_data: Dict[str, Any],
    settings: Settings
):
    """
    Asynchronous WhatsApp message processing function.

    This function handles the actual RAG processing and response sending
    in the background, allowing the webhook to respond quickly to Meta.

    Args:
        message_data: Parsed message data from WhatsApp
        settings: Application settings
    """
    logger.info("WhatsApp async processing started", message_data_keys=list(message_data.keys()))
    try:
        # Extract message details
        user_phone = message_data["from"]
        user_message = message_data["text"]
        message_id = message_data["message_id"]

        logger.info(
            "Processing WhatsApp message asynchronously",
            from_number=user_phone,
            message_id=message_id,
            message_length=len(user_message),
            message_text=user_message[:100]  # Log first 100 chars for debugging
        )

        # Process message through RAG system (same as web UI)
        rag_service = get_rag_service()
        logger.info(
            "WhatsApp RAG processing started",
            query=user_message,
            conversation_id=f"whatsapp_{user_phone}",
            max_context_chunks=8,
            confidence_threshold=settings.confidence_threshold
        )

        rag_response = await rag_service.generate_response(
            query=user_message,
            conversation_id=f"whatsapp_{user_phone}",
            max_context_chunks=8,  # Same as web UI for consistent responses
            confidence_threshold=settings.confidence_threshold,  # Use same threshold as web UI
            channel="chat"  # Use same channel as web UI for consistent processing
        )

        logger.info(
            "WhatsApp RAG processing completed",
            response_length=len(rag_response.get("response", "")),
            confidence=rag_response.get("confidence", 0),
            sources_count=len(rag_response.get("sources", [])),
            response_preview=rag_response.get("response", "")[:100]
        )

        # Send RAG response back via WhatsApp
        whatsapp_client = get_whatsapp_client()
        logger.info(
            "Sending WhatsApp response",
            to=user_phone,
            response_to_send=rag_response.get("response", "")[:100]
        )

        send_result = await whatsapp_client.send_rag_response(
            to=user_phone,
            rag_response=rag_response,
            include_sources=False  # Keep WhatsApp messages clean and concise
        )

        logger.info(
            "WhatsApp response sent",
            to=user_phone,
            send_success=send_result.get("success", False),
            message_id=send_result.get("message_id", "unknown")
        )

        logger.info(
            "WhatsApp RAG response sent successfully",
            to=user_phone,
            message_id=send_result.get("message_id"),
            confidence=rag_response.get("confidence"),
            processing_time_ms=rag_response.get("processing_time_ms")
        )

    except Exception as e:
        logger.error(
            "Async WhatsApp message processing failed",
            error=str(e),
            from_number=message_data.get("from"),
            message_id=message_data.get("message_id")
        )


@app.get("/whatsapp/status")
async def whatsapp_status(settings: Settings = Depends(get_settings)):
    """
    Get WhatsApp integration status and health.

    Returns:
        WhatsApp integration status and configuration
    """
    try:
        if not settings.is_whatsapp_configured():
            return JSONResponse(content={
                "status": "not_configured",
                "configured": False,
                "message": "WhatsApp Business API credentials not configured"
            })

        whatsapp_client = get_whatsapp_client()
        health_status = await whatsapp_client.health_check()

        return JSONResponse(content={
            "status": "configured",
            "configured": True,
            "health": health_status,
            "phone_number_id": settings.whatsapp_phone_number_id,
            "webhook_url": settings.get_webhook_url()
        })

    except Exception as e:
        logger.error("WhatsApp status check failed", error=str(e))
        return JSONResponse(content={
            "status": "error",
            "configured": settings.is_whatsapp_configured(),
            "error": str(e)
        })
