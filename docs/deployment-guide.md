# SAIA-RAG Deployment Guide

Step-by-step instructions for deploying SAIA-RAG in development and production.

## 🚀 **Quick Start (Development)**

### **Prerequisites**
- Docker and Docker Compose installed
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git (for cloning the repository)

### **1. Clone and Setup**
```bash
# Clone the repository
git clone <repository-url>
cd SAIA-RAG

# Copy environment template
cp .env.example .env
```

### **2. Configure Environment**
Edit `.env` file with your settings:
```bash
# Required: Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Customize other settings
TENANT_ID=t_customerA
ENVIRONMENT=development
DEBUG=true
```

### **3. Start Development Environment**
```bash
# Start all services
docker compose -f docker-compose.dev.yml up -d

# Check service status
docker compose -f docker-compose.dev.yml ps

# View logs
docker compose -f docker-compose.dev.yml logs -f
```

### **4. Verify Installation**
```bash
# Test API health (comprehensive status)
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "ok",
#   "service": "SAIA-RAG API",
#   "version": "0.1.0",
#   "timestamp": "2025-08-23T17:33:09.882406",
#   "environment": "development",
#   "dependencies": {}
# }

# Test root endpoint
curl http://localhost:8000/

# Test Qdrant health
curl http://localhost:6333/readyz

# Access API documentation (development only)
open http://localhost:8000/docs
open http://localhost:8000/redoc
```

### **5. Stop Services**
```bash
# Stop all services
docker compose -f docker-compose.dev.yml down

# Stop and remove volumes (clean slate)
docker compose -f docker-compose.dev.yml down -v
```

## 🏭 **Production Deployment**

### **Prerequisites**
- Docker and Docker Compose
- Domain name with DNS configured
- Production OpenAI API key

### **Production Setup**
1. **Create `.env.prod`**:
```bash
ENVIRONMENT=production
DEBUG=false
OPENAI_API_KEY=sk-your-production-key
API_KEY=your-secure-api-key
CORS_ORIGINS=["https://yourdomain.com"]
```

2. **Create `docker-compose.prod.yml`** (with Caddy reverse proxy)
3. **Deploy**:
```bash
docker build -t saia-rag:latest .
docker compose -f docker-compose.prod.yml up -d
```

## 🚨 **Troubleshooting**

### **Common Issues**
- **Service won't start**: Check logs with `docker compose logs`
- **API returns 500**: Check application logs with `docker logs saia-rag-api-dev`
- **Qdrant connection issues**: Test health with `curl http://localhost:6333/readyz`

### **Health Checks**
```bash
# Check all services
docker compose ps

# Test API with comprehensive health info
curl http://localhost:8000/health | python3 -m json.tool

# Test root endpoint
curl http://localhost:8000/ | python3 -m json.tool

# Test Qdrant
curl http://localhost:6333/readyz
```

### **Monitoring and Logs**
```bash
# View application logs (structured JSON in production)
docker logs saia-rag-api-dev

# Follow logs in real-time
docker logs -f saia-rag-api-dev

# View Qdrant logs
docker logs saia-rag-qdrant-dev

# Check service resource usage
docker stats
```

**Log Features**:
- **Structured Logging**: JSON format with timestamps and context
- **Request Logging**: All API requests logged with method, path, and status
- **Error Logging**: Comprehensive error context with stack traces
- **Health Check Logging**: Service health status changes logged

---

For detailed production configuration, see the development rules and configuration documentation.
