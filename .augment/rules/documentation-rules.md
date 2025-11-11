---
type: "always_apply"
---

# SAIA-RAG Documentation Rules

**CRITICAL**: These documentation rules are MANDATORY for all AI agents working on the SAIA-RAG project.

## 📋 MANDATORY DOCUMENTATION WORKFLOW

### Documentation-First Development (NON-NEGOTIABLE)
1. **ALWAYS** update documentation BEFORE making code changes
2. **NEVER** commit code without updating relevant documentation
3. **VERIFY** documentation accuracy after implementation
4. **MAINTAIN** documentation that stays current with codebase

### Required Documentation Updates
- **New Features**: Update README.md and API specification
- **Configuration Changes**: Update configuration.md
- **Breaking Changes**: Update deployment guide with migration notes
- **Bug Fixes**: Update troubleshooting sections
- **Dependencies**: Update requirements and compatibility notes

## 📝 CODE DOCUMENTATION STANDARDS

### Python Docstring Format (MANDATORY)
```python
def process_document(content: str, chunk_size: int = 1000) -> list[DocumentChunk]:
    """
    Process document content into chunks for vector storage.

    Args:
        content: Raw document content to process
        chunk_size: Maximum size of each chunk in characters

    Returns:
        List of DocumentChunk objects with content and metadata

    Raises:
        ValueError: If content is empty or chunk_size is invalid

    Example:
        >>> chunks = process_document("Sample content")
        >>> len(chunks) > 0
        True
    """
```

### Module Documentation (MANDATORY)
```python
"""
SAIA-RAG Document Processing Module

Provides functionality for processing uploaded documents,
including text extraction, chunking, and preparation for vector storage.

Key Components:
    - DocumentProcessor: Main class for document processing
    - process_document(): Function for document chunking

Usage:
    from app.document_processor import process_document
    chunks = process_document("content")
"""
```

### Inline Comments (REQUIRED FOR COMPLEX LOGIC)
```python
# Calculate overlap indices to ensure continuity between chunks
start_idx = max(0, i * (chunk_size - overlap))

# Skip chunks that are too small to be meaningful
if end_idx - start_idx < MIN_CHUNK_SIZE:
    continue
```

## 🌐 API DOCUMENTATION STANDARDS

### FastAPI Endpoint Documentation (MANDATORY)
```python
@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process chat query",
    description="Submit a query to the RAG system for contextual response"
)
async def chat(request: ChatRequest):
    """Process chat request with RAG context."""
```

### Request/Response Examples (REQUIRED)
```python
# In docstring or OpenAPI description:
"""
Example Request:
{
    "message": "How do I reset my password?",
    "conversation_id": "conv_123"
}

Example Response:
{
    "response": "To reset your password, follow these steps...",
    "confidence": 0.85,
    "sources": ["doc_456"]
}
"""
```

## 📁 REQUIRED DOCUMENTATION FILES

### Core Documentation (MUST EXIST)
- `README.md` - Project overview, quick start, current status
- `docs/architecture.md` - System design and component interaction
- `docs/configuration.md` - Environment variables and Pydantic v2 settings
- `docs/api-specification.md` - API endpoints with examples
- `docs/deployment-guide.md` - Setup instructions for dev/prod

### Documentation Structure (IMMUTABLE)
```
SAIA-RAG/
├── README.md              # Main project documentation
├── docs/
│   ├── architecture.md    # System design
│   ├── configuration.md   # Config and environment variables
│   ├── api-specification.md # API documentation
│   └── deployment-guide.md # Setup and deployment
└── .augment/rules/
    ├── dev-rules.md       # Development patterns and standards
    └── documentation-rules.md # This file
```

## 📋 DOCUMENTATION TEMPLATES

### README.md Structure (MANDATORY)
```markdown
# Project Title
Brief description and badges

## Architecture Overview
Simple diagram and tech stack

## Quick Start
Prerequisites, setup, and verification steps

## Current Status
What's complete vs. in progress

## Configuration
Key environment variables

## Documentation Links
Links to other docs

## Contributing
Reference to development rules
```

### API Endpoint Template (MANDATORY)
```markdown
#### **Endpoint Name**
```http
POST /endpoint
```

**Description**: What this endpoint does

**Request**:
```json
{"field": "value"}
```

**Response**:
```json
{"result": "success"}
```

**Example**:
```bash
curl -X POST http://localhost:8000/endpoint \
  -H "Content-Type: application/json" \
  -d '{"field": "value"}'
```
```

### Configuration Documentation Template (MANDATORY)
```markdown
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VAR_NAME` | `string` | `default` | What this variable controls |
```

## 🔄 DOCUMENTATION MAINTENANCE

### Update Triggers (MANDATORY)
- **Code Changes**: Update relevant docs immediately
- **New Endpoints**: Add to API specification
- **Config Changes**: Update configuration.md
- **Breaking Changes**: Update deployment guide
- **Bug Fixes**: Update troubleshooting sections

### Quality Checklist (VERIFY BEFORE COMMIT)
- [ ] All code examples are tested and working
- [ ] Links to other documentation are valid
- [ ] Formatting is consistent
- [ ] Examples include error scenarios
- [ ] Version information is current

### Documentation Review (MANDATORY)
- **With every code change**: Verify docs are updated
- **Before releases**: Comprehensive documentation review
- **Monthly**: Check for broken links and outdated examples

## 🚫 FORBIDDEN PRACTICES

### Documentation Violations (NEVER DO)
- ❌ Committing code without updating relevant documentation
- ❌ Creating documentation that duplicates existing content
- ❌ Writing documentation without working examples
- ❌ Using outdated or incorrect code examples
- ❌ Creating overly complex documentation structures

### Complexity Violations (NEVER DO)
- ❌ Creating multiple documentation files for the same topic
- ❌ Over-engineering documentation with unnecessary detail
- ❌ Duplicating information across multiple files
- ❌ Creating documentation that's harder to maintain than the code

## ✅ DOCUMENTATION DECISION TREE

### When Adding New Feature
1. Is it a new API endpoint? → Update api-specification.md
2. Does it change configuration? → Update configuration.md
3. Does it affect deployment? → Update deployment-guide.md
4. Does it change architecture? → Update architecture.md
5. Is it user-facing? → Update README.md

### When Fixing Bugs
1. Is it a configuration issue? → Update configuration.md troubleshooting
2. Is it a deployment issue? → Update deployment-guide.md
3. Is it an API issue? → Update api-specification.md error handling
4. Is it a setup issue? → Update README.md quick start

### When Changing Dependencies
1. Update requirements.txt version constraints
2. Update README.md prerequisites if needed
3. Update deployment-guide.md if setup changes
4. Update configuration.md if new config options

## 📊 DOCUMENTATION QUALITY METRICS

### Required Standards
- **Accuracy**: All examples must work as documented
- **Completeness**: All public APIs and configs documented
- **Clarity**: Documentation must be understandable by new developers
- **Consistency**: Follow established templates and patterns
- **Currency**: Documentation stays current with code changes

### Success Criteria
- New developers can set up the project using only documentation
- All API endpoints have working examples
- All configuration options are documented with examples
- Breaking changes include migration instructions
- Documentation builds and links work correctly

---

**REMEMBER**: Simple, accurate, current documentation is better than complex, comprehensive documentation that becomes outdated. Keep it practical and maintainable.
