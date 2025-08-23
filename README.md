# SAIA-RAG: Customer Support AI Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/latest/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.12+-purple.svg)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)

A production-ready RAG (Retrieval-Augmented Generation) chatbot for customer support built with FastAPI, Qdrant vector database, and OpenAI.

## 🏗️ **Architecture Overview**

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
    │   Client  │
    └───────────┘
```

### **Technology Stack**
- **Backend**: FastAPI (Python 3.11+) with Pydantic v2
- **Vector Database**: Qdrant (Docker container)
- **LLM Provider**: OpenAI (gpt-4o-mini, text-embedding-3-large)
- **Containerization**: Docker Compose
- **Reverse Proxy**: Caddy (production)

## 🚀 **Quick Start**

### **Prerequisites**
- Docker and Docker Compose
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### **1. Clone and Setup**
```bash
git clone <repository-url>
cd SAIA-RAG
cp .env.example .env
```

### **2. Configure Environment**
Edit `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
TENANT_ID=t_customerA
```

### **3. Start Services**
```bash
# Start development environment
docker compose -f docker-compose.dev.yml up -d

# Check service health
curl http://localhost:8000/health
curl http://localhost:6333/readyz
```

### **4. Access Services**
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📋 **Current Status**

### ✅ **Complete**
- [x] **Foundation**: Project structure, dependencies, Docker setup
- [x] **Configuration**: Pydantic v2 settings with validation
- [x] **Basic API**: FastAPI app with health endpoint

### 🚧 **In Progress**
- [ ] **Core Application**: Middleware, error handling, logging
- [ ] **Vector Store**: Qdrant integration and operations
- [ ] **API Endpoints**: Document upload, RAG chat, admin features

## 🔧 **Configuration**

The application uses Pydantic v2 for robust configuration management:

```python
# Core settings loaded from environment variables
ENVIRONMENT=development          # development, production, testing
OPENAI_API_KEY=sk-...           # Required: OpenAI API key
TENANT_ID=t_customerA           # Tenant identifier
QDRANT_URL=http://qdrant:6333   # Vector database URL
CONFIDENCE_THRESHOLD=0.35       # RAG confidence threshold
```

See [Configuration Documentation](docs/configuration.md) for complete details.

## 📖 **Documentation**

- [**Architecture Guide**](docs/architecture.md) - System design and components
- [**API Specification**](docs/api-specification.md) - API endpoints and examples
- [**Configuration Guide**](docs/configuration.md) - Environment variables and settings
- [**Deployment Guide**](docs/deployment-guide.md) - Setup instructions

## 🐳 **Docker Services**

### **Development Environment**
```bash
# Start all services
docker compose -f docker-compose.dev.yml up -d

# View logs
docker compose -f docker-compose.dev.yml logs -f

# Stop services
docker compose -f docker-compose.dev.yml down
```

### **Service Health Checks**
- **FastAPI**: `curl http://localhost:8000/health`
- **Qdrant**: `curl http://localhost:6333/readyz`

## 🔒 **Security**

- **API Key Authentication**: Required in production, optional in development
- **CORS Configuration**: Environment-specific allowed origins
- **Input Validation**: Pydantic v2 models with comprehensive validation
- **Rate Limiting**: Configurable request limits per endpoint
- **Environment Isolation**: Separate configurations for dev/prod

## 🧪 **Testing**

```bash
# Run tests inside container
docker exec saia-rag-api-dev pytest

# Run specific test file
docker exec saia-rag-api-dev pytest tests/test_config.py

# Run with coverage
docker exec saia-rag-api-dev pytest --cov=app
```

## 📊 **Monitoring**

- **Health Endpoints**: `/health` for service status
- **Structured Logging**: JSON logs with correlation IDs
- **Metrics**: Prometheus-compatible metrics (planned)
- **Observability**: Request tracing and performance monitoring (planned)

## 🤝 **Contributing**

1. **Follow Development Rules**: See [`.augment/rules/dev-rules.md`](.augment/rules/dev-rules.md)
2. **Follow Documentation Rules**: See [`.augment/rules/documentation-rules.md`](.augment/rules/documentation-rules.md)
3. **Clean Implementation**: Research first, build incrementally, zero technical debt
4. **Documentation**: Update docs with all changes
5. **Testing**: Write tests for all new functionality

---

**Built with ❤️ using FastAPI, Qdrant, and OpenAI**
