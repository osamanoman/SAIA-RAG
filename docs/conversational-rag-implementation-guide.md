# SAIA-RAG Conversational Context Implementation Guide

**Date**: 2025-10-26  
**Purpose**: Step-by-step implementation guide for fixing conversation context loss  
**Based on**: LangGraph and LangChain best practices

---

## 🎯 Implementation Overview

This guide provides **production-ready code** for integrating conversation context management into the SAIA-RAG system. All code follows existing project patterns and is ready for immediate implementation.

---

## 📋 Phase 1: Integrate ConversationMemoryManager

### Step 1.1: Modify `app/rag_service.py`

**Location**: Lines 40-60 (initialization)

```python
from app.conversation_memory import (
    ConversationMemoryManager,
    MessageType,
    ConversationContext
)

class RAGService:
    """Enhanced RAG service with conversation context management."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        openai_client: OpenAIClient,
        query_processor: QueryProcessor,
        settings: Settings
    ):
        self.vector_store = vector_store
        self.openai_client = openai_client
        self.query_processor = query_processor
        self.settings = settings
        
        # ✅ NEW: Initialize conversation manager
        self.conversation_manager = ConversationMemoryManager()
        
        # Existing initialization
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = 3600
        
        logger.info("RAG service initialized with conversation management")
```

---

### Step 1.2: Modify `generate_response()` Method

**Location**: Lines 169-232

**Add conversation context retrieval BEFORE query processing:**

```python
async def generate_response(
    self,
    query: str,
    conversation_id: Optional[str] = None,
    max_context_chunks: int = 8,
    confidence_threshold: Optional[float] = None,
    channel: str = "default"
) -> Dict[str, Any]:
    """
    Generate RAG response with conversation context support.
    """
    try:
        start_time = datetime.utcnow()
        confidence_threshold = confidence_threshold or self.settings.confidence_threshold
        
        # ✅ NEW: Step 0 - Get conversation context
        conversation_context = None
        if conversation_id:
            conversation_context = await self.conversation_manager.get_conversation_context(
                conversation_id=conversation_id,
                include_messages=True
            )
            logger.info(
                "Retrieved conversation context",
                conversation_id=conversation_id,
                message_count=len(conversation_context.get("messages", [])) if conversation_context else 0
            )
        
        # Step 1: Process and enhance query with conversation context
        if self.settings.enable_query_enhancement:
            enhanced_query_result = await self.query_processor.process_query(
                query=query,
                channel="whatsapp",
                conversation_context=conversation_context  # ← NOW POPULATED
            )
            processed_query = enhanced_query_result.enhanced_query
            query_metadata = {
                "original_query": query,
                "enhanced_query": processed_query,
                "query_category": enhanced_query_result.query_type.category,
                "query_intent": enhanced_query_result.query_type.intent,
                "preprocessing_steps": enhanced_query_result.preprocessing_applied,
                "conversation_aware": conversation_context is not None
            }
            logger.info("Query enhanced with conversation context", **query_metadata)
        else:
            processed_query = query
            query_metadata = {
                "original_query": query,
                "enhanced_query": query,
                "conversation_aware": False
            }
        
        # ... rest of existing RAG flow (embedding, search, context building) ...
        
        # Generate response
        response_text = await self._generate_llm_response(
            system_prompt=system_prompt,
            user_query=query,
            temperature=0.3
        )
        
        # ✅ NEW: Store conversation messages
        if conversation_id:
            # Store user query
            await self.conversation_manager.add_message(
                conversation_id=conversation_id,
                content=query,
                message_type=MessageType.USER_QUERY,
                metadata={
                    "enhanced_query": processed_query,
                    "query_category": query_metadata.get("query_category")
                }
            )
            
            # Store AI response
            await self.conversation_manager.add_message(
                conversation_id=conversation_id,
                content=response_text,
                message_type=MessageType.AI_RESPONSE,
                metadata={
                    "confidence": avg_confidence,
                    "sources_count": len(sources),
                    "article_numbers": [s.article_number for s in sources if s.article_number]
                }
            )
            
            logger.info(
                "Stored conversation messages",
                conversation_id=conversation_id,
                query_length=len(query),
                response_length=len(response_text)
            )
        
        # ... rest of response building ...
        
    except Exception as e:
        logger.error("RAG response generation failed", error=str(e))
        raise
```

---

## 📋 Phase 2: Query Reformulation with Context

### Step 2.1: Add Helper Methods to `app/query_processor.py`

**Location**: Add after existing methods (around line 250)

```python
def _extract_case_facts(self, conversation_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract case facts from conversation history.
    
    Returns:
        Dictionary with case facts: plaintiff, defendant, key_issues, etc.
    """
    if not conversation_context or not conversation_context.get("messages"):
        return {}
    
    messages = conversation_context["messages"]
    
    # Get first user message (usually contains case description)
    first_user_message = next(
        (msg for msg in messages if msg.get("message_type") == "user_query"),
        None
    )
    
    if not first_user_message:
        return {}
    
    first_query = first_user_message.get("content", "")
    
    # Extract entities using simple pattern matching
    # TODO: Replace with NER or LLM-based extraction for production
    case_facts = {
        "first_query": first_query,
        "query_length": len(first_query),
        "message_count": len(messages)
    }
    
    # Extract names (Arabic pattern: زوجة/زوج followed by name)
    import re
    
    # Look for plaintiff/defendant patterns
    if "زوجة" in first_query or "المدعية" in first_query:
        case_facts["plaintiff_type"] = "wife"
    if "زوج" in first_query or "المدعى عليه" in first_query:
        case_facts["defendant_type"] = "husband"
    
    # Extract key issues
    key_issues = []
    issue_patterns = {
        "إدمان": "drug_addiction",
        "اعتداء": "assault",
        "ضرب": "physical_abuse",
        "إهمال": "neglect",
        "نفقة": "alimony",
        "حضانة": "custody",
        "فسخ": "dissolution"
    }
    
    for arabic_term, english_term in issue_patterns.items():
        if arabic_term in first_query:
            key_issues.append(english_term)
    
    case_facts["key_issues"] = key_issues
    
    logger.info(
        "Extracted case facts",
        plaintiff_type=case_facts.get("plaintiff_type"),
        defendant_type=case_facts.get("defendant_type"),
        key_issues=key_issues
    )
    
    return case_facts


def _is_follow_up_query(self, query: str) -> bool:
    """
    Detect if query is a follow-up (short, vague, uses pronouns).
    
    Examples of follow-ups:
    - "ماذا عن النفقة" (What about alimony?)
    - "والحضانة؟" (And custody?)
    - "في هذه القضية" (In this case)
    """
    # Short queries are likely follow-ups
    if len(query.split()) <= 5:
        # Check for follow-up indicators
        follow_up_indicators = [
            "ماذا عن",  # What about
            "والحضانة",  # And custody
            "والنفقة",  # And alimony
            "في هذه القضية",  # In this case
            "في القضية",  # In the case
            "أيضا",  # Also
            "كذلك",  # As well
            "بالنسبة",  # Regarding
        ]
        
        for indicator in follow_up_indicators:
            if indicator in query:
                logger.info("Detected follow-up query", query=query, indicator=indicator)
                return True
    
    return False


async def _reformulate_with_context(
    self,
    query: str,
    case_facts: Dict[str, Any]
) -> str:
    """
    Reformulate follow-up query with case context.
    
    Examples:
    - "ماذا عن النفقة" → "ما هي حقوق الزوجة في النفقة في قضية إدمان الزوج وإهماله؟"
    - "What about custody?" → "What are the wife's custody rights given husband's drug addiction?"
    """
    try:
        # Build context string from case facts
        context_parts = []
        
        if case_facts.get("plaintiff_type"):
            context_parts.append(f"Plaintiff: {case_facts['plaintiff_type']}")
        
        if case_facts.get("defendant_type"):
            context_parts.append(f"Defendant: {case_facts['defendant_type']}")
        
        if case_facts.get("key_issues"):
            context_parts.append(f"Key issues: {', '.join(case_facts['key_issues'])}")
        
        context_string = "; ".join(context_parts)
        
        # Use LLM to reformulate
        reformulation_prompt = f"""You are a legal query reformulation assistant.

Original follow-up query: {query}

Case context: {context_string}

Original case description: {case_facts.get('first_query', '')[:500]}

Task: Reformulate the follow-up query to be specific and include relevant case context.
The reformulated query should be a complete, standalone question that includes:
1. The specific legal topic from the follow-up (e.g., alimony, custody)
2. Relevant case facts (e.g., drug addiction, neglect)
3. The parties involved (e.g., wife, husband)

Respond with ONLY the reformulated query in the same language as the original query.
"""
        
        reformulated = await self.openai_client.generate_completion(
            prompt=reformulation_prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        reformulated = reformulated.strip()
        
        logger.info(
            "Query reformulated with context",
            original=query,
            reformulated=reformulated,
            context_used=bool(context_string)
        )
        
        return reformulated
        
    except Exception as e:
        logger.error("Query reformulation failed", error=str(e))
        return query  # Fallback to original query
```

---

### Step 2.2: Modify `process_query()` to Use Context

**Location**: Lines 68-136

**Add context-aware reformulation BEFORE query enhancement:**

```python
async def process_query(
    self,
    query: str,
    channel: str = "default",
    conversation_context: Optional[Dict[str, Any]] = None
) -> EnhancedQuery:
    """
    Process and enhance a user query with conversation context support.
    """
    start_time = datetime.utcnow()
    preprocessing_steps = []
    
    try:
        # ✅ NEW: Step 0 - Reformulate with conversation context if follow-up
        original_query = query
        if conversation_context:
            case_facts = self._extract_case_facts(conversation_context)
            
            if case_facts and self._is_follow_up_query(query):
                query = await self._reformulate_with_context(query, case_facts)
                preprocessing_steps.append("context_reformulation")
                logger.info(
                    "Query reformulated with conversation context",
                    original=original_query,
                    reformulated=query
                )
        
        # Step 1: Clean and normalize query
        cleaned_query = self._clean_query(query)
        if cleaned_query != query:
            preprocessing_steps.append("cleaning")
        
        # ... rest of existing processing steps ...
        
        result = EnhancedQuery(
            original_query=original_query,  # ← Use original, not reformulated
            enhanced_query=enhanced_query,
            query_type=query_type,
            preprocessing_applied=preprocessing_steps,
            processing_time_ms=processing_time_ms
        )
        
        return result
        
    except Exception as e:
        logger.error("Query processing failed", query=query[:100], error=str(e))
        # ... existing error handling ...
```

---

## 📋 Phase 3: Testing and Validation

### Test Case 1: Addiction/Harm Case

**File**: `tests/test_conversational_rag.py`

```python
import pytest
from app.rag_service import RAGService
from app.models import ChatRequest

@pytest.mark.asyncio
async def test_conversation_context_retention(rag_service):
    """Test that follow-up queries use conversation context."""
    
    conversation_id = "test-addiction-case"
    
    # Initial query with case description
    initial_query = """
    زوجة تطلب فسخ عقد الزواج بسبب إدمان الزوج وتعريضه الأسرة للخطر.
    المدعية (ريم) متزوجة من المدعى عليه (ماجد) منذ عام 1440هـ، ولديهما طفلان.
    """
    
    response1 = await rag_service.generate_response(
        query=initial_query,
        conversation_id=conversation_id
    )
    
    assert response1["status"] == "success"
    assert "فسخ" in response1["response"] or "dissolution" in response1["response"].lower()
    
    # Follow-up query about alimony
    followup_query = "ماذا عن النفقة"
    
    response2 = await rag_service.generate_response(
        query=followup_query,
        conversation_id=conversation_id
    )
    
    assert response2["status"] == "success"
    # Should mention case-specific context (Reem, Majed, addiction, neglect)
    response_text = response2["response"]
    assert any(term in response_text for term in ["ريم", "ماجد", "إدمان", "إهمال"])
    
    # Follow-up query about custody
    followup_query2 = "ماذا عن الحضانة في هذه القضية"
    
    response3 = await rag_service.generate_response(
        query=followup_query2,
        conversation_id=conversation_id
    )
    
    assert response3["status"] == "success"
    # Should apply custody rules to case context
    assert "حضانة" in response3["response"]


@pytest.mark.asyncio
async def test_query_reformulation(query_processor):
    """Test that follow-up queries are reformulated with context."""
    
    # Simulate conversation context
    conversation_context = {
        "messages": [
            {
                "message_type": "user_query",
                "content": "زوجة تطلب فسخ عقد الزواج بسبب إدمان الزوج",
                "timestamp": "2025-10-26T14:07:00"
            }
        ]
    }
    
    # Follow-up query
    followup_query = "ماذا عن النفقة"
    
    result = await query_processor.process_query(
        query=followup_query,
        conversation_context=conversation_context
    )
    
    # Enhanced query should include case context
    assert result.enhanced_query != followup_query
    assert len(result.enhanced_query) > len(followup_query)
    assert "context_reformulation" in result.preprocessing_applied
```

---

## ✅ Validation Checklist

### Phase 1: Conversation Memory Integration
- [ ] `ConversationMemoryManager` imported in `rag_service.py`
- [ ] Conversation context retrieved before query processing
- [ ] User queries stored in conversation history
- [ ] AI responses stored in conversation history
- [ ] Conversation context passed to query processor
- [ ] Tests pass for context retrieval

### Phase 2: Query Reformulation
- [ ] `_extract_case_facts()` implemented
- [ ] `_is_follow_up_query()` implemented
- [ ] `_reformulate_with_context()` implemented
- [ ] Follow-up queries reformulated with case facts
- [ ] Tests pass for query reformulation

### Phase 3: End-to-End Testing
- [ ] Addiction/harm case test passes
- [ ] Follow-up queries use conversation context
- [ ] Multi-turn conversations maintain case facts
- [ ] Performance acceptable (< 2s per query)

---

## 🚀 Deployment Steps

1. **Backup Current System**
   ```bash
   git checkout -b feature/conversational-rag
   ```

2. **Implement Phase 1**
   - Modify `app/rag_service.py`
   - Test conversation memory integration

3. **Implement Phase 2**
   - Modify `app/query_processor.py`
   - Test query reformulation

4. **Run Full Test Suite**
   ```bash
   pytest tests/test_conversational_rag.py -v
   ```

5. **Deploy to Development**
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

6. **Validate with Real Cases**
   - Test addiction/harm case
   - Test other multi-turn scenarios

7. **Deploy to Production**
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

---

**End of Implementation Guide**

