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

**Description**: Comprehensive service health check with dependency monitoring

**Response**:
```json
{
  "status": "ok",
  "service": "SAIA-RAG API",
  "version": "0.1.0",
  "timestamp": "2025-08-23T17:33:09.882406",
  "environment": "development",
  "dependencies": {}
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service or dependencies are unhealthy

**Example**:
```bash
curl -X GET http://localhost:8000/health
```

**Response Details**:
- `status`: Overall service health status
- `service`: Service name identifier
- `version`: Current application version
- `timestamp`: ISO timestamp of health check
- `environment`: Current environment (development/production)
- `dependencies`: Health status of external dependencies (future: Qdrant, OpenAI)

#### **Root Endpoint**
```http
GET /
```

**Description**: Service information with environment-aware features

**Response (Development)**:
```json
{
  "message": "SAIA-RAG Customer Support AI Assistant",
  "status": "running",
  "version": "0.1.0",
  "environment": "development",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

**Response (Production)**:
```json
{
  "message": "SAIA-RAG Customer Support AI Assistant",
  "status": "running",
  "version": "0.1.0",
  "environment": "production"
}
```

**Example**:
```bash
curl -X GET http://localhost:8000/
```

**Features**:
- **Environment-Aware**: Documentation links only shown in development
- **Version Information**: Current application version included
- **Status Indicator**: Service running status

## 🚨 **Error Handling**

### **Error Response Format**
All errors follow a consistent format with proper logging:

```json
{
  "error": {
    "code": 404,
    "message": "Not Found",
    "path": "/nonexistent"
  }
}
```

### **Common Status Codes**
- `200 OK`: Request successful
- `404 Not Found`: Endpoint not found
- `500 Internal Server Error`: Server error (logged with full context)
- `503 Service Unavailable`: Service health check failed

### **Error Logging**
- All errors are logged with structured logging (JSON format)
- HTTP exceptions include request path, method, and status code
- General exceptions include full error context and stack traces
- No sensitive information exposed in error responses

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
