"""
SAIA-RAG FastAPI Application

Main application entry point with health checks, error handling, and logging.
Follows clean architecture patterns with proper dependency injection.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
import structlog

from .config import get_settings, Settings

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
        # Basic service health
        health_data = {
            "status": "ok",
            "service": "SAIA-RAG API",
            "version": settings.app_version,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.environment,
            "dependencies": {}
        }

        # TODO: Add dependency health checks when implemented
        # - Qdrant vector database connectivity
        # - OpenAI API connectivity

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
