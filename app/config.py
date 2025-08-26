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
        default="Wazen AI Assistant",
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
    openai_max_tokens: int = Field(
        default=500,
        alias="OPENAI_MAX_TOKENS",
        description="Maximum tokens for OpenAI responses"
    )
    openai_temperature: float = Field(
        default=0.7,
        alias="OPENAI_TEMPERATURE",
        description="Temperature for OpenAI response generation"
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
        default=0.25,  # Lowered for better customer support coverage
        alias="CONFIDENCE_THRESHOLD",
        description="Minimum confidence threshold for RAG responses"
    )
    max_search_results: int = Field(
        default=8,
        alias="MAX_SEARCH_RESULTS",
        description="Maximum number of search results to retrieve"
    )
    chunk_size: int = Field(
        default=800,  # Optimized for customer support content
        alias="CHUNK_SIZE",
        description="Document chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=200,
        alias="CHUNK_OVERLAP",
        description="Overlap between document chunks"
    )
    rag_search_limit: int = Field(
        default=8,
        alias="RAG_SEARCH_LIMIT",
        description="Maximum chunks to retrieve for RAG context"
    )

    # === CUSTOMER SUPPORT SPECIFIC CONFIGURATION ===
    escalation_threshold: float = Field(
        default=0.40,
        alias="ESCALATION_THRESHOLD",
        description="Confidence threshold below which to suggest human escalation"
    )
    whatsapp_confidence_threshold: float = Field(
        default=0.20,
        alias="WHATSAPP_CONFIDENCE_THRESHOLD",
        description="Lower confidence threshold for WhatsApp responses"
    )
    max_response_tokens: int = Field(
        default=300,
        alias="MAX_RESPONSE_TOKENS",
        description="Maximum tokens for generated responses"
    )
    enable_query_enhancement: bool = Field(
        default=True,
        alias="ENABLE_QUERY_ENHANCEMENT",
        description="Enable query preprocessing and enhancement"
    )
    enable_conversation_memory: bool = Field(
        default=True,
        alias="ENABLE_CONVERSATION_MEMORY",
        description="Enable conversation context memory for multi-turn interactions"
    )
    support_categories: list[str] = Field(
        default=["troubleshooting", "billing", "setup", "general", "policies"],
        alias="SUPPORT_CATEGORIES",
        description="Customer support content categories for routing"
    )
    
    # === API CONFIGURATION ===
    api_key: Optional[str] = Field(
        default=None,
        alias="API_KEY",
        description="API key for authentication (optional in development)"
    )
    base_url: str = Field(
        default="http://localhost:8000",
        alias="BASE_URL",
        description="Base URL for the application (used for webhooks and external references)"
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

    @field_validator("escalation_threshold")
    @classmethod
    def validate_escalation_threshold(cls, v: float) -> float:
        """Validate escalation threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Escalation threshold must be between 0.0 and 1.0")
        return v

    @field_validator("whatsapp_confidence_threshold")
    @classmethod
    def validate_whatsapp_confidence_threshold(cls, v: float) -> float:
        """Validate WhatsApp confidence threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("WhatsApp confidence threshold must be between 0.0 and 1.0")
        return v

    @field_validator("max_response_tokens")
    @classmethod
    def validate_max_response_tokens(cls, v: int) -> int:
        """Validate maximum response tokens."""
        if v < 50 or v > 2000:
            raise ValueError("Max response tokens must be between 50 and 2000")
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

    def get_webhook_url(self) -> str:
        """Get the full WhatsApp webhook URL."""
        return f"{self.base_url.rstrip('/')}/whatsapp/webhook"

    def should_escalate(self, confidence: float) -> bool:
        """Check if response should be escalated to human based on confidence."""
        return confidence < self.escalation_threshold

    def get_confidence_threshold_for_channel(self, channel: str = "default") -> float:
        """Get appropriate confidence threshold based on channel."""
        if channel.lower() == "whatsapp":
            return self.whatsapp_confidence_threshold
        return self.confidence_threshold

    def is_support_category_valid(self, category: str) -> bool:
        """Check if a support category is valid."""
        return category.lower() in [cat.lower() for cat in self.support_categories]

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

    @field_validator("whatsapp_verify_token")
    @classmethod
    def validate_whatsapp_verify_token(cls, v: Optional[str]) -> Optional[str]:
        """Validate WhatsApp verify token."""
        if v is not None and len(v) < 10:
            raise ValueError("WhatsApp verify token must be at least 10 characters long")
        return v

    @field_validator("whatsapp_phone_number_id")
    @classmethod
    def validate_whatsapp_phone_number_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate WhatsApp phone number ID."""
        if v is not None and not v.isdigit():
            raise ValueError("WhatsApp phone number ID must contain only digits")
        return v

    @field_validator("whatsapp_access_token")
    @classmethod
    def validate_whatsapp_access_token(cls, v: Optional[str]) -> Optional[str]:
        """Validate WhatsApp access token."""
        if v is not None and len(v) < 50:
            raise ValueError("WhatsApp access token appears to be too short")
        return v


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
