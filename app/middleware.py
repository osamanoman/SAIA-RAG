"""
SAIA-RAG Middleware Module

Middleware stack for FastAPI application including CORS, authentication,
rate limiting, logging, and security headers.
"""

import time
import hashlib
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog

from .config import get_settings

# Get logger
logger = structlog.get_logger()

# Rate limiting storage (in-memory for now, consider Redis for production)
_rate_limit_storage: Dict[str, Dict[str, Any]] = defaultdict(dict)


def setup_middleware(app):
    """Setup all middleware in the correct order."""
    
    # 1. Trusted Host (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure based on your domain
    )
    
    # 2. CORS (cross-origin requests)
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # 3. Custom middleware for rate limiting and logging
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(logging_middleware)
    app.middleware("http")(security_headers_middleware)
    
    logger.info("Middleware stack configured successfully")


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware with different limits for different endpoints."""
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Different rate limits for different endpoints
    if request.url.path.startswith("/whatsapp/webhook"):
        # WhatsApp webhook: 10 requests per minute per IP
        limit = 10
        window = 60
        endpoint_type = "whatsapp_webhook"
    elif request.url.path.startswith("/chat"):
        # Chat endpoint: 30 requests per minute per IP
        limit = 30
        window = 60
        endpoint_type = "chat"
    elif request.url.path.startswith("/documents/upload"):
        # Document upload: 5 requests per minute per IP
        limit = 5
        window = 60
        endpoint_type = "document_upload"
    else:
        # Default: 100 requests per minute per IP
        limit = 100
        window = 60
        endpoint_type = "default"
    
    # Check rate limit
    if not _check_rate_limit(client_ip, endpoint_type, limit, window):
        logger.warning(
            "Rate limit exceeded",
            client_ip=client_ip,
            endpoint=request.url.path,
            endpoint_type=endpoint_type,
            limit=limit,
            window=window
        )
        return Response(
            content='{"error": "Rate limit exceeded. Please try again later."}',
            status_code=429,
            media_type="application/json"
        )
    
    # Continue with request
    response = await call_next(request)
    return response


def _check_rate_limit(client_ip: str, endpoint_type: str, limit: int, window: int) -> bool:
    """Check if client is within rate limit."""
    
    current_time = datetime.utcnow()
    key = f"{client_ip}:{endpoint_type}"
    
    # Get current rate limit data
    rate_data = _rate_limit_storage[key]
    
    # Clean old entries
    if "requests" in rate_data:
        rate_data["requests"] = [
            req_time for req_time in rate_data["requests"]
            if current_time - req_time < timedelta(seconds=window)
        ]
    else:
        rate_data["requests"] = []
    
    # Check if limit exceeded
    if len(rate_data["requests"]) >= limit:
        return False
    
    # Add current request
    rate_data["requests"].append(current_time)
    return True


async def logging_middleware(request: Request, call_next):
    """Logging middleware for request/response tracking."""
    
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time_ms=round(process_time * 1000, 2)
    )
    
    return response


async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Remove server info
    if "server" in response.headers:
        del response.headers["server"]
    
    return response


def get_rate_limit_info(client_ip: str, endpoint_type: str) -> Dict[str, Any]:
    """Get rate limit information for a client and endpoint."""
    
    key = f"{client_ip}:{endpoint_type}"
    rate_data = _rate_limit_storage.get(key, {})
    
    if "requests" in rate_data:
        current_time = datetime.utcnow()
        # Count requests in last minute
        recent_requests = [
            req_time for req_time in rate_data["requests"]
            if current_time - req_time < timedelta(seconds=60)
        ]
        
        return {
            "client_ip": client_ip,
            "endpoint_type": endpoint_type,
            "requests_last_minute": len(recent_requests),
            "total_requests": len(rate_data["requests"])
        }
    
    return {
        "client_ip": client_ip,
        "endpoint_type": endpoint_type,
        "requests_last_minute": 0,
        "total_requests": 0
    }


def clear_rate_limit_storage():
    """Clear rate limit storage (useful for testing)."""
    global _rate_limit_storage
    _rate_limit_storage.clear()
    logger.info("Rate limit storage cleared")
