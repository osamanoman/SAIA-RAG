"""
SAIA-RAG OpenAI Client Module

OpenAI API integration for embeddings and chat completion.
Provides centralized OpenAI operations with error handling and logging.
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

import openai
from openai import OpenAI
import structlog

from .config import get_settings, Settings

# Get logger
logger = structlog.get_logger()


class OpenAIClient:
    """
    OpenAI API client with embedding and chat completion capabilities.
    
    Provides centralized OpenAI operations following clean architecture patterns:
    - Embedding generation for document chunks and queries
    - Chat completion for RAG responses
    - Error handling and retry logic
    - Usage tracking and logging
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize OpenAI client with configuration.

        Args:
            settings: Application settings with OpenAI configuration
        """
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=30.0,  # 30 second timeout
            max_retries=3   # Retry failed requests up to 3 times
        )

        logger.info(
            "OpenAI client initialized",
            chat_model=settings.openai_chat_model,
            embed_model=settings.openai_embed_model,
            embed_dim=settings.embed_dim,
            timeout=30.0,
            max_retries=3
        )
    
    async def generate_embeddings(
        self, 
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Optional model override (uses config default if None)
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        if not texts:
            return []
        
        model = model or self.settings.openai_embed_model
        
        try:
            start_time = datetime.utcnow()
            
            # Generate embeddings using OpenAI API
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            # Extract embedding vectors
            embeddings = [data.embedding for data in response.data]
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(
                "Embeddings generated successfully",
                texts_count=len(texts),
                model=model,
                embed_dim=len(embeddings[0]) if embeddings else 0,
                processing_time_ms=processing_time_ms,
                tokens_used=response.usage.total_tokens
            )
            
            return embeddings
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                "Failed to generate embeddings",
                texts_count=len(texts),
                model=model,
                error=str(e),
                error_type=error_type
            )

            # Provide specific error messages for common issues
            if "rate_limit" in str(e).lower():
                raise Exception("OpenAI API rate limit exceeded. Please try again later.")
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                raise Exception("OpenAI API key is invalid or missing. Please check configuration.")
            elif "quota" in str(e).lower():
                raise Exception("OpenAI API quota exceeded. Please check your billing.")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            model: Optional model override
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text], model)
        return embeddings[0] if embeddings else []
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override (uses config default if None)
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-2.0)
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Chat completion response with content and metadata
            
        Raises:
            Exception: If chat completion fails
        """
        model = model or self.settings.openai_chat_model
        max_tokens = max_tokens or 500
        temperature = temperature if temperature is not None else 0.0  # Deterministic responses
        
        try:
            start_time = datetime.utcnow()
            
            # Prepare messages with optional system prompt
            chat_messages = []
            
            if system_prompt:
                chat_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            chat_messages.extend(messages)
            
            # Generate chat completion with deterministic parameters
            response = self.client.chat.completions.create(
                model=model,
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=42  # Fixed seed for deterministic responses
            )
            
            # Extract response content
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = {
                "content": content,
                "finish_reason": finish_reason,
                "model": model,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "processing_time_ms": processing_time_ms
            }
            
            logger.info(
                "Chat completion generated successfully",
                model=model,
                messages_count=len(chat_messages),
                tokens_used=response.usage.total_tokens,
                processing_time_ms=processing_time_ms,
                finish_reason=finish_reason
            )
            
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                "Failed to generate chat completion",
                model=model,
                messages_count=len(messages),
                error=str(e),
                error_type=error_type
            )

            # Provide specific error messages for common issues
            if "rate_limit" in str(e).lower():
                raise Exception("OpenAI API rate limit exceeded. Please try again later.")
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                raise Exception("OpenAI API key is invalid or missing. Please check configuration.")
            elif "quota" in str(e).lower():
                raise Exception("OpenAI API quota exceeded. Please check your billing.")
            elif "context_length" in str(e).lower():
                raise Exception("Message too long for the model. Please shorten your request.")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check OpenAI API connectivity and health.
        
        Returns:
            Health status dictionary
        """
        try:
            start_time = datetime.utcnow()
            
            # Test with a simple embedding request
            test_response = self.client.embeddings.create(
                model=self.settings.openai_embed_model,
                input=["health check test"]
            )
            
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            health_data = {
                "status": "healthy",
                "response_time_ms": response_time_ms,
                "chat_model": self.settings.openai_chat_model,
                "embed_model": self.settings.openai_embed_model,
                "embed_dim": len(test_response.data[0].embedding),
                "api_key_configured": bool(self.settings.openai_api_key)
            }
            
            logger.info("OpenAI health check successful", **health_data)
            return health_data
            
        except Exception as e:
            error_data = {
                "status": "unhealthy",
                "error": str(e),
                "chat_model": self.settings.openai_chat_model,
                "embed_model": self.settings.openai_embed_model,
                "api_key_configured": bool(self.settings.openai_api_key)
            }
            logger.error("OpenAI health check failed", **error_data)
            return error_data


@lru_cache()
def get_openai_client() -> OpenAIClient:
    """
    Get cached OpenAI client instance.
    
    Uses @lru_cache to ensure client is initialized only once
    and reused throughout the application lifecycle.
    
    Returns:
        Cached OpenAI client instance
    """
    settings = get_settings()
    return OpenAIClient(settings)


# Export for easy importing
__all__ = ["OpenAIClient", "get_openai_client"]
