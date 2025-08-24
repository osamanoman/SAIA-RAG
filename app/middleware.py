"""
SAIA-RAG Middleware Module

All middleware implementations following development rules.
Middleware order (last added = first executed):
1. CORS (outermost)
2. Rate limiting
3. Logging
4. Security headers (innermost)
"""

import time
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .config import get_settings

# Get logger
logger = structlog.get_logger()


def setup_middleware(app) -> None:
    """
    Setup all middleware in the correct order per dev rules.

    Order (last added = first executed):
    1. CORS (outermost)
    2. Rate limiting
    3. Logging
    4. Security headers (innermost)

    Args:
        app: FastAPI application instance
    """
    settings = get_settings()

    # 4. Security Headers Middleware (innermost - last executed)
    app.middleware("http")(security_headers)

    # 3. Logging Middleware
    app.middleware("http")(logging_middleware)

    # 2. Rate Limiting Middleware
    app.middleware("http")(rate_limiter)

    # 1. CORS Middleware (outermost - first executed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    logger.info("Middleware setup completed", environment=settings.environment)





async def rate_limiter(request: Request, call_next: Callable) -> Response:
    """Rate limiting middleware."""
    # TODO: Implement proper rate limiting
    # For now, just pass through
    response = await call_next(request)
    return response


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Request/response logging middleware."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    process_time_ms = int(process_time * 1000)
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time_ms=process_time_ms
    )
    
    return response


async def security_headers(request: Request, call_next: Callable) -> Response:
    """Security headers middleware."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add process time header
    if "X-Process-Time" not in response.headers:
        response.headers["X-Process-Time"] = "0"
    
    return response
