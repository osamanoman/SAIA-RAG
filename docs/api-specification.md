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

### **✅ RAG Operations (Fully Implemented)**

#### **Chat with RAG**
```http
POST /chat
```

**Description**: Process chat query using RAG (Retrieval-Augmented Generation)

**Authentication**: Required in production

**Request**:
```json
{
  "message": "What insurance services does Wazen provide?",
  "conversation_id": "test-001"
}
```

**Response**:
```json
{
  "status": "success",
  "timestamp": "2025-08-24T09:20:30.753818",
  "response": "Wazen provides comprehensive car insurance services in Saudi Arabia, which include:\n\n1. Vehicle protection\n2. Accident coverage\n3. Theft protection\n4. 24/7 customer support\n\nThey offer competitive rates and fast claim processing for all types of vehicles, including sedans, SUVs, and motorcycles. The coverage options include third-party liability, comprehensive coverage, and collision protection.",
  "conversation_id": "test-001",
  "confidence": 0.0,
  "sources": [],
  "processing_time_ms": 2109,
  "tokens_used": 931
}
```

#### **Search Documents**
```http
POST /search
```

**Description**: Search document chunks using vector similarity

**Request**:
```json
{
  "query": "car insurance coverage",
  "limit": 5
}
```

**Response**:
```json
{
  "status": "success",
  "timestamp": "2025-08-24T09:21:59.967358",
  "results": [
    {
      "chunk_id": "b33a72ac-734a-4a6e-8682-ca72fc48f084_chunk_0",
      "document_id": "b33a72ac-734a-4a6e-8682-ca72fc48f084",
      "title": "Wazen Insurance Services",
      "content": "Wazen provides comprehensive car insurance services...",
      "score": 0.43825454,
      "metadata": {
        "title": "Wazen Insurance Services",
        "category": "insurance",
        "tags": ["wazen", "insurance", "cars", "saudi arabia"],
        "author": "Wazen Team"
      }
    }
  ],
  "total_results": 2,
  "processing_time_ms": 707,
  "query": "car insurance coverage"
}
```

### **✅ System Management (Fully Implemented)**

#### **Escalate to Human Support**
```http
POST /escalate
```

**Authentication**: Required in production

**Request**:
```json
{
  "conversation_id": "test-001",
  "reason": "complex_technical_issue",
  "user_message": "I need help with a complex insurance claim",
  "context": "User needs specialized assistance"
}
```

**Response**:
```json
{
  "status": "escalated",
  "timestamp": "2025-08-24T09:23:35.019186",
  "escalation_id": "3adff1b0-0cc4-4393-b542-930fafd8f488",
  "ticket_number": "SAIA-3ADFF1B0",
  "estimated_response_time": "1-2 business days"
}
```

#### **Submit Feedback**
```http
POST /feedback
```

**Authentication**: Required in production

**Request**:
```json
{
  "conversation_id": "test-001",
  "rating": 5,
  "category": "helpfulness",
  "comment": "Great response about insurance services!"
}
```

**Response**:
```json
{
  "status": "received",
  "timestamp": "2025-08-24T09:24:42.950100",
  "feedback_id": "5bfaa275-6758-44b5-b528-927ac7e80c7b",
  "message": "Thank you for your positive feedback! We're glad we could help."
}
```

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
