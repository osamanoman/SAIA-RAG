---
type: "always_apply"
---

# SAIA-RAG AI Agent Development Rules

**CRITICAL**: These rules are MANDATORY for all AI agents working on the SAIA-RAG project. Deviation from these patterns is strictly prohibited.

## 🏗️ SYSTEM ARCHITECTURE (IMMUTABLE)

### Core Technology Stack
- **Backend**: FastAPI (Python 3.11+) - NO other web frameworks
- **Vector Database**: Qdrant (Docker container) - NO other vector stores
- **LLM Provider**: OpenAI (gpt-4o-mini, text-embedding-3-large/small) - NO other providers
- **Reverse Proxy**: Caddy (production only) - NO nginx or Apache
- **Containerization**: Docker Compose - ALL services MUST run in containers

### Container Architecture (NEVER CHANGE)
```
┌─────────────────────────────────────────┐
│              Docker Host                │
│  ┌─────────────┐    ┌─────────────┐    │
│  │   FastAPI   │◄──►│   Qdrant    │    │
│  │ Container   │    │ Container   │    │
│  │ :8000       │    │ :6333       │    │
│  └─────────────┘    └─────────────┘    │
│         │                               │
└─────────┼───────────────────────────────┘
          │ (Production: via Caddy)
    ┌─────▼─────┐
    │   Host    │
    └───────────┘
```

## 📁 PROJECT STRUCTURE (IMMUTABLE)

```
SAIA-RAG/
├── app/                    # FastAPI application (NEVER rename)
│   ├── __init__.py
│   ├── main.py            # Main FastAPI app (NEVER rename)
│   ├── config.py          # Pydantic Settings (NEVER rename)
│   ├── models.py          # Pydantic models (NEVER rename)
│   ├── middleware.py      # All middleware (NEVER rename)
│   ├── vector_store.py    # Qdrant operations (NEVER rename)
│   ├── utils.py           # Utility functions (NEVER rename)
│   ├── ingest.py          # Document processing (ADD when needed)
│   ├── retrieve.py        # RAG retrieval (ADD when needed)
│   └── fallbacks.py       # Fallback logic (ADD when needed)
├── tests/                 # Test files (REQUIRED)
├── scripts/               # Setup/utility scripts
├── data/                  # Local data (Docker volumes)
├── docs/                  # Documentation
├── docker-compose.dev.yml # Development (NEVER rename)
├── docker-compose.prod.yml# Production (NEVER rename)
├── Dockerfile             # API container (NEVER rename)
├── Caddyfile             # Reverse proxy config
├── requirements.txt       # Python deps (NEVER rename)
├── .env                  # Development environment variables
├── .env.prod             # Production environment variables
├── .env.dev              # Dev environment template (optional)
└── .env.prod.sample      # Prod environment template (optional)
```

## 🔒 MANDATORY API ENDPOINTS

### Required Endpoints (NEVER CHANGE)
```python
@app.get("/health")           # Health check - IMMUTABLE
@app.post("/ingest")          # Document ingestion - IMMUTABLE
@app.post("/chat")            # RAG chat - IMMUTABLE
@app.post("/escalate")        # Human escalation - IMMUTABLE
@app.post("/feedback")        # Response feedback - IMMUTABLE
```

### Endpoint Implementation Rules
1. **ALWAYS** use existing Pydantic models from `app/models.py`
2. **ALWAYS** include API key authentication via `Depends(api_key_auth)`
3. **ALWAYS** use structured logging with `structlog`
4. **ALWAYS** include proper error handling with HTTPException
5. **NEVER** create new response models without updating `app/models.py`

### Example (CORRECT):
```python
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: Optional[str] = Depends(api_key_auth)
):
    logger.info("Chat request received", message=request.message[:100])
    try:
        # Implementation here
        return ChatResponse(...)
    except Exception as e:
        logger.error("Chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail="Chat request failed")
```

## 🗄️ DATABASE & VECTOR STORE RULES

### Qdrant Configuration (IMMUTABLE)
- **Collection naming**: `docs_{tenant_id}` (e.g., `docs_t_customerA`)
- **Vector dimensions**: 3072 (text-embedding-3-large) or 1536 (text-embedding-3-small)
- **Distance metric**: COSINE (NEVER change)
- **Connection URL**: `http://qdrant:6333` (Docker internal network)

### Vector Store Operations (MANDATORY PATTERNS)
```python
# CORRECT: Always use the global vector store instance
from app.vector_store import get_vector_store

vector_store = get_vector_store()
results = await vector_store.search_similar(query_vector, limit=8)

# WRONG: Never instantiate QdrantVectorStore directly
# vector_store = QdrantVectorStore()  # FORBIDDEN
```

## 🔧 CONFIGURATION MANAGEMENT

### Environment Files (IMMUTABLE STRUCTURE)
- **`.env`** - Development environment variables (local development)
- **`.env.prod`** - Production environment variables (production deployment)
- **`.env.dev`** - Development template (optional, for documentation)
- **`.env.prod.sample`** - Production template (optional, for documentation)

### Environment Variables (STRICT NAMING)
```bash
# REQUIRED (NEVER change names)
OPENAI_API_KEY=sk-...
TENANT_ID=t_customerA
QDRANT_URL=http://qdrant:6333

# OPTIONAL (NEVER change names)
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
EMBED_DIM=3072
CONFIDENCE_THRESHOLD=0.35
ENVIRONMENT=development  # or production
```

### Environment File Rules
1. **NEVER** commit `.env` or `.env.prod` to version control
2. **ALWAYS** use `.env` for local development
3. **ALWAYS** use `.env.prod` for production deployment
4. **KEEP** template files (`.env.dev`, `.env.prod.sample`) for documentation
5. **LOAD** environment files in Pydantic settings with `env_file=".env"`

### Settings Usage (MANDATORY PATTERN)
```python
# CORRECT: Always use the global settings instance
from app.config import get_settings
settings = get_settings()

# WRONG: Never instantiate Settings directly
# settings = Settings()  # FORBIDDEN
```

## 📦 DEPENDENCY MANAGEMENT (MANDATORY)

### Requirements.txt Structure (IMMUTABLE)
```txt
# Core Framework
fastapi>=0.115.0,<0.116.0
uvicorn[standard]>=0.30.0,<0.31.0

# Pydantic v2 and Settings
pydantic>=2.0.0,<3.0.0
pydantic-settings>=2.0.0,<3.0.0

# Vector Database
qdrant-client>=1.12.0,<2.0.0

# OpenAI Integration
openai>=1.0.0,<2.0.0

# Document Processing
python-multipart>=0.0.9,<1.0.0

# Utilities
structlog>=24.0.0,<25.0.0
python-dotenv>=1.0.0,<2.0.0

# Development and Testing
pytest>=8.0.0,<9.0.0
pytest-asyncio>=0.24.0,<1.0.0
httpx>=0.27.0,<1.0.0
black>=24.0.0,<25.0.0
ruff>=0.6.0,<1.0.0
```

### Version Pinning Rules
1. **ALWAYS** use version ranges (not exact pins)
2. **PIN** major versions to avoid breaking changes
3. **VERIFY** compatibility between dependencies
4. **TEST** all dependencies work together before committing

## 🛡️ SECURITY & MIDDLEWARE (IMMUTABLE ORDER)

### Middleware Stack (NEVER CHANGE ORDER)
```python
# Last added = First executed (IMMUTABLE ORDER)
app.middleware("http")(security_headers)    # 4th (innermost)
app.middleware("http")(logging_middleware)  # 3rd
app.middleware("http")(rate_limiter)        # 2nd
app.middleware("http")(cors_middleware)     # 1st (outermost)
```

### Authentication Rules
- **Development**: API key optional (controlled by `settings.debug`)
- **Production**: API key MANDATORY for all endpoints except `/health`
- **NEVER** bypass authentication in production
- **ALWAYS** use `Depends(api_key_auth)` for protected endpoints

## 📝 PYDANTIC V2 MODELS (STRICT PATTERNS)

### Pydantic v2 Key Changes (CRITICAL)
- Use `pydantic_settings.BaseSettings` (not `pydantic.BaseSettings`)
- Use `@field_validator` with `@classmethod` decorator (not `@validator`)
- Use `model_config = ConfigDict(...)` (not `Config` class)
- Use `Field(..., alias="ENV_VAR")` for environment variable mapping
- Use `info.data.get()` in validators (not `values.get()`)

### Model Organization (IMMUTABLE)
- **ALL** models MUST be in `app/models.py`
- **NEVER** define models in other files
- **ALWAYS** inherit from BaseModel
- **ALWAYS** include field descriptions
- **ALWAYS** use Pydantic v2 syntax

### Pydantic v2 Example (CORRECT):
```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()
```

## 🐳 DOCKER RULES (IMMUTABLE)

### Development Environment
```yaml
# docker-compose.dev.yml (NEVER change service names)
services:
  qdrant:    # IMMUTABLE name
    image: qdrant/qdrant:v1.12.1  # Pin version
    ports: ["6333:6333"]

  api:       # IMMUTABLE name
    build: .
    ports: ["8000:8000"]
    depends_on:
      qdrant: {condition: service_healthy}
```

### Container Communication (MANDATORY)
- FastAPI → Qdrant: `http://qdrant:6333` (NEVER use localhost)
- **NEVER** expose Qdrant ports in production
- **ALWAYS** use health checks for service dependencies

## 🧪 TESTING REQUIREMENTS (MANDATORY)

### Test Structure (REQUIRED)
```
tests/
├── test_api.py           # API endpoint tests
├── test_vector_store.py  # Qdrant operations tests
├── test_models.py        # Pydantic model tests
└── test_integration.py   # End-to-end tests
```

### Testing Rules
1. **ALWAYS** write tests for new endpoints
2. **ALWAYS** test Pydantic model validation
3. **ALWAYS** test vector store operations
4. **NEVER** commit code without tests
5. **ALWAYS** use pytest fixtures for common setup

## 🚫 FORBIDDEN PRACTICES

### Architecture Violations (NEVER DO)
- ❌ Using any web framework other than FastAPI
- ❌ Using any vector database other than Qdrant
- ❌ Using any LLM provider other than OpenAI
- ❌ Running services outside Docker containers
- ❌ Changing the established project structure

### Code Violations (NEVER DO)
- ❌ Defining Pydantic models outside `app/models.py`
- ❌ Bypassing the global settings instance
- ❌ Creating direct Qdrant client instances
- ❌ Changing middleware order
- ❌ Hardcoding configuration values

### Docker Violations (NEVER DO)
- ❌ Changing service names in docker-compose files
- ❌ Using localhost URLs between containers
- ❌ Exposing unnecessary ports
- ❌ Running services on host instead of containers

## ✅ DECISION TREES

### Adding New Endpoint
1. Is it one of the 5 required endpoints? → Use existing implementation
2. Is it a new endpoint? → Add to `app/main.py` following patterns
3. Need new models? → Add to `app/models.py` only
4. Need authentication? → Use `Depends(api_key_auth)`
5. Need logging? → Use `structlog.get_logger()`

### Adding New Functionality
1. Document processing? → Add to `app/ingest.py`
2. RAG retrieval? → Add to `app/retrieve.py`
3. Fallback logic? → Add to `app/fallbacks.py`
4. Utility functions? → Add to `app/utils.py`
5. Configuration? → Add to `app/config.py`

### Vector Store Operations
1. Need to search? → Use `client.search(collection_name, query_vector, limit)`
2. Need to store? → Use `client.upsert(collection_name, points)`
3. Need collection info? → Use `client.get_collection(collection_name)`
4. **ALWAYS** use global QdrantClient instance pattern
5. **NEVER** create new QdrantClient instances per request

## 📚 AUTHORITATIVE CODE EXAMPLES

### Current Implementation Status
**FOUNDATION COMPLETE**: The following files represent the current clean implementation:
- `requirements.txt` - Verified compatible dependencies ✅
- `Dockerfile` - Python 3.11 container setup ✅
- `docker-compose.dev.yml` - Working container orchestration ✅
- `app/main.py` - Basic FastAPI application ✅

### Files to be Built (Following Established Patterns)
- `app/config.py` - Pydantic v2 settings management
- `app/models.py` - Pydantic v2 model definitions
- `app/middleware.py` - Middleware implementation
- `app/vector_store.py` - Qdrant client operations

**CRITICAL**: Any deviation from these established patterns is a violation of project standards and must be corrected immediately.

## 🔍 CODE QUALITY STANDARDS

### Python Code Style (MANDATORY)
```bash
# ALWAYS run before committing
black .                    # Code formatting
ruff check --fix .        # Linting and fixes
```

### Import Organization (STRICT ORDER)
```python
# 1. Standard library imports
import os
import sys
from typing import List, Dict, Optional

# 2. Third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

# 3. Local application imports
from .config import get_settings
from .models import ChatRequest, ChatResponse
from .vector_store import get_vector_store
```

### Error Handling (MANDATORY PATTERN)
```python
# CORRECT: Always use this pattern
try:
    # Operation here
    result = await some_operation()
    logger.info("Operation successful", result_count=len(result))
    return result
except SpecificException as e:
    logger.error("Specific error occurred", error=str(e), context="additional_info")
    raise HTTPException(status_code=400, detail="Specific error message")
except Exception as e:
    logger.error("Unexpected error", error=str(e), operation="operation_name")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Logging Standards (IMMUTABLE)
```python
# CORRECT: Always use structured logging
logger = structlog.get_logger()
logger.info("Operation started", user_id="123", operation="chat")
logger.error("Operation failed", error=str(e), user_id="123")

# WRONG: Never use print or basic logging
# print("Debug message")  # FORBIDDEN
# logging.info("Message")  # FORBIDDEN
```

## 🔐 SECURITY REQUIREMENTS (NON-NEGOTIABLE)

### API Key Handling
```python
# CORRECT: Always validate API keys
@app.post("/protected-endpoint")
async def protected_endpoint(
    request: SomeRequest,
    api_key: Optional[str] = Depends(api_key_auth)  # MANDATORY
):
    # Implementation
```

### Input Validation (MANDATORY)
```python
# CORRECT: Always validate and sanitize inputs
class UserInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

    @validator("message")
    def sanitize_message(cls, v):
        # Remove dangerous characters, validate format
        return clean_text(v)  # Use utility function
```

### Environment Variables (SECURE HANDLING)
```python
# CORRECT: Never log sensitive values
logger.info("Config loaded",
           openai_model=settings.openai_chat_model,
           # NEVER log: openai_api_key=settings.openai_api_key
           )

# Use sanitize_for_logging() for dictionaries with mixed data
safe_data = sanitize_for_logging(request_data)
logger.info("Request data", **safe_data)
```

## 📊 PERFORMANCE REQUIREMENTS

### Database Operations (MANDATORY PATTERNS)
```python
# CORRECT: Always use connection pooling and async operations
vector_store = get_vector_store()  # Reuse global instance
results = await vector_store.search_similar(query_vector, limit=8)

# WRONG: Never create new connections per request
# client = QdrantClient(url=settings.qdrant_url)  # FORBIDDEN
```

### Memory Management
```python
# CORRECT: Process large files in chunks
async def process_large_document(file_content: bytes):
    chunk_size = settings.chunk_size
    for i in range(0, len(file_content), chunk_size):
        chunk = file_content[i:i + chunk_size]
        await process_chunk(chunk)

# WRONG: Never load entire large files into memory
# content = await file.read()  # DANGEROUS for large files
```

## 🚀 DEPLOYMENT REQUIREMENTS

### Environment-Specific Configurations
```python
# CORRECT: Always check environment
if settings.is_production():
    # Production-specific logic
    app.docs_url = None  # Disable docs in production
    app.redoc_url = None
else:
    # Development-specific logic
    app.docs_url = "/docs"
```

### Health Check Implementation (MANDATORY)
```python
@app.get("/health")
async def health_check():
    # ALWAYS check dependencies
    vector_health = await vector_store.health_check()

    status = "ok" if vector_health["status"] == "healthy" else "degraded"

    return {
        "status": status,
        "timestamp": datetime.utcnow(),
        "version": settings.app_version,
        "dependencies": {"vector_store": vector_health}
    }
```

## 🎯 FINAL ENFORCEMENT RULES

### Before Any Code Changes
1. **READ** existing code patterns in the target file
2. **FOLLOW** the exact same patterns and naming conventions
3. **USE** existing imports and dependencies
4. **VALIDATE** against these rules before implementation
5. **TEST** the changes with existing test patterns

### Code Review Checklist
- [ ] Uses existing Pydantic models from `app/models.py`
- [ ] Follows established import order and naming
- [ ] Includes proper error handling and logging
- [ ] Uses global instances (settings, vector_store)
- [ ] Maintains Docker container architecture
- [ ] Includes appropriate tests
- [ ] Follows security requirements

### Violation Response
If any AI agent violates these rules:
1. **STOP** immediately
2. **REVERT** the violating changes
3. **IMPLEMENT** using the correct patterns
4. **VERIFY** compliance with these rules

**REMEMBER**: These rules exist to maintain code quality, security, and architectural consistency. They are not suggestions—they are requirements.

**DOCUMENTATION**: All documentation requirements are defined in `.augment/rules/documentation-rules.md` and MUST be followed.
