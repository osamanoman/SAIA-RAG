"""
SAIA-RAG Configuration Module

Pydantic v2 settings management with environment variable support.
Follows clean architecture patterns with proper validation and type safety.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings using Pydantic v2.
    
    Loads configuration from environment variables with proper validation,
    type conversion, and default values.
    """
    
    # === CORE APPLICATION SETTINGS ===
    app_name: str = Field(
        default="SAIA-RAG Customer Support AI Assistant",
        description="Application name"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    environment: str = Field(
        default="development",
        alias="ENVIRONMENT",
        description="Environment: development, production"
    )
    debug: bool = Field(
        default=True,
        alias="DEBUG",
        description="Enable debug mode"
    )
    
    # === OPENAI CONFIGURATION ===
    openai_api_key: str = Field(
        ...,
        alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM and embeddings"
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_CHAT_MODEL",
        description="OpenAI chat model to use"
    )
    openai_embed_model: str = Field(
        default="text-embedding-3-large",
        alias="OPENAI_EMBED_MODEL",
        description="OpenAI embedding model to use"
    )
    
    # === VECTOR DATABASE CONFIGURATION ===
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        alias="QDRANT_URL",
        description="Qdrant vector database URL"
    )
    embed_dim: int = Field(
        default=3072,
        alias="EMBED_DIM",
        description="Embedding dimensions (3072 for text-embedding-3-large)"
    )
    
    # === TENANT CONFIGURATION ===
    tenant_id: str = Field(
        default="t_customerA",
        alias="TENANT_ID",
        description="Tenant identifier for multi-tenancy"
    )
    
    # === RAG CONFIGURATION ===
    confidence_threshold: float = Field(
        default=0.35,
        alias="CONFIDENCE_THRESHOLD",
        description="Minimum confidence threshold for RAG responses"
    )
    max_search_results: int = Field(
        default=8,
        alias="MAX_SEARCH_RESULTS",
        description="Maximum number of search results to retrieve"
    )
    chunk_size: int = Field(
        default=1000,
        alias="CHUNK_SIZE",
        description="Document chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=200,
        alias="CHUNK_OVERLAP",
        description="Overlap between document chunks"
    )
    
    # === API CONFIGURATION ===
    api_key: Optional[str] = Field(
        default=None,
        alias="API_KEY",
        description="API key for authentication (optional in development)"
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="CORS_ORIGINS",
        description="Allowed CORS origins"
    )
    
    # === PYDANTIC V2 MODEL CONFIGURATION ===
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # === FIELD VALIDATORS (PYDANTIC V2 SYNTAX) ===
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = {"development", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key appears to be too short")
        return v
    
    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate confidence threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("embed_dim")
    @classmethod
    def validate_embed_dim(cls, v: int) -> int:
        """Validate embedding dimensions."""
        allowed_dims = {1536, 3072}  # text-embedding-3-small, text-embedding-3-large
        if v not in allowed_dims:
            raise ValueError(f"Embedding dimensions must be one of: {allowed_dims}")
        return v
    
    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        """Validate tenant ID format."""
        if not v.startswith("t_"):
            raise ValueError("Tenant ID must start with 't_'")
        if len(v) < 3:
            raise ValueError("Tenant ID must be at least 3 characters")
        return v
    
    # === UTILITY METHODS ===
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == "testing"
    
    def get_collection_name(self) -> str:
        """Get the Qdrant collection name for this tenant."""
        return f"docs_{self.tenant_id}"

    # === WHATSAPP BUSINESS API CONFIGURATION ===
    whatsapp_access_token: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_ACCESS_TOKEN",
        description="WhatsApp Business API access token"
    )
    whatsapp_phone_number_id: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_PHONE_NUMBER_ID",
        description="WhatsApp Business phone number ID"
    )
    whatsapp_verify_token: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_VERIFY_TOKEN",
        description="WhatsApp webhook verify token"
    )
    whatsapp_business_account_id: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_BUSINESS_ACCOUNT_ID",
        description="WhatsApp Business account ID (optional)"
    )
    whatsapp_app_id: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_APP_ID",
        description="WhatsApp App ID (optional)"
    )
    whatsapp_app_secret: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_APP_SECRET",
        description="WhatsApp App secret (optional)"
    )

    def is_whatsapp_configured(self) -> bool:
        """
        Check if WhatsApp Business API is properly configured.

        Returns:
            True if minimum required fields are configured
        """
        return all([
            self.whatsapp_access_token,
            self.whatsapp_phone_number_id,
            self.whatsapp_verify_token
        ])


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses @lru_cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    """
    return Settings()


# Export for easy importing
__all__ = ["Settings", "get_settings"]
