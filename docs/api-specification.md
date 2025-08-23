# SAIA-RAG API Specification

API documentation for the SAIA-RAG Customer Support AI Assistant.

## 🌐 **Base Information**

- **Base URL**: `http://localhost:8000` (development)
- **Content Type**: `application/json`
- **Authentication**: API Key (optional in development, required in production)

## 🔒 **Authentication**

### **API Key Authentication**
```http
Authorization: Bearer your-api-key-here
```

**Development**: API key is optional
**Production**: API key is required for all endpoints except `/health`

## 📋 **Current API Endpoints**

### **✅ Implemented Endpoints**

#### **Health Check**
```http
GET /health
```

**Description**: Check service health and status

**Response**:
```json
{
  "status": "ok",
  "service": "SAIA-RAG API",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "dependencies": {
    "vector_store": {
      "status": "healthy",
      "response_time_ms": 12
    }
  }
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service or dependencies are unhealthy

**Example**:
```bash
curl -X GET http://localhost:8000/health
```

#### **Root Endpoint**
```http
GET /
```

**Description**: API information and welcome message

**Response**:
```json
{
  "message": "SAIA-RAG Customer Support AI Assistant",
  "status": "running",
  "docs": "/docs"
}
```

## 📋 **Planned API Endpoints**

### **Document Management**
- `POST /documents/upload` - Upload and process documents
- `GET /documents` - List uploaded documents
- `DELETE /documents/{id}` - Delete document

### **RAG Operations**
- `POST /chat` - Submit query for RAG response
- `POST /search` - Search document chunks

### **System Management**
- `POST /escalate` - Escalate to human support
- `POST /feedback` - Submit response feedback
- `GET /admin/stats` - System statistics (admin only)

## 📝 **Interactive Documentation**

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 **Testing**

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with authentication (when implemented)
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/chat \
     -d '{"message": "Hello"}'
```

---

This API specification will be updated as endpoints are implemented.
