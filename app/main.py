"""
SAIA-RAG FastAPI Application
Main application entry point with basic health check.
"""

from fastapi import FastAPI

# Create FastAPI application
app = FastAPI(
    title="SAIA-RAG Customer Support AI Assistant",
    description="RAG-based customer support chatbot with smart fallbacks",
    version="0.1.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "ok",
        "service": "SAIA-RAG API",
        "version": "0.1.0"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAIA-RAG Customer Support AI Assistant",
        "status": "running",
        "docs": "/docs"
    }
