"""
SAIA-RAG Service Module

Complete RAG (Retrieval-Augmented Generation) implementation.
Orchestrates document processing, vector search, and response generation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import hashlib
from functools import lru_cache

import structlog

from .config import get_settings
from .openai_client import get_openai_client
from .vector_store import get_vector_store
from .document_processor import get_document_processor
from .query_processor import get_query_processor
# ESCALATION SYSTEM REMOVED
from .response_formatter import get_response_formatter
from .models import SourceDocument

# Get logger
logger = structlog.get_logger()


class RAGService:
    """
    Complete RAG service implementation.
    
    Provides:
    - Document ingestion with embedding generation
    - Context retrieval from vector store
    - Response generation with source citations
    - Conversation context management
    """
    
    def __init__(self):
        """Initialize RAG service with dependencies."""
        self.settings = get_settings()
        self.openai_client = get_openai_client()
        self.vector_store = get_vector_store()
        self.document_processor = get_document_processor()
        self.query_processor = get_query_processor()
        # ESCALATION MANAGER REMOVED
        self.response_formatter = get_response_formatter()

        # Simple in-memory cache for frequent queries
        self._query_cache = {}
        self._cache_max_size = 100

        logger.info("RAG service initialized", cache_max_size=self._cache_max_size)
    
    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document content
            metadata: Document metadata
            content_type: MIME type of content
            
        Returns:
            Ingestion result with document ID and chunk count
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Process document into chunks
            processing_result = self.document_processor.process_document(
                content, metadata, content_type
            )
            
            if not processing_result["success"]:
                return {
                    "success": False,
                    "error": processing_result["error"],
                    "document_id": document_id
                }
            
            chunks = processing_result["chunks"]
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.openai_client.generate_embeddings(chunk_texts)
            
            # Index chunks in vector store
            index_result = self.vector_store.index_document_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings
            )
            
            end_time = datetime.utcnow()
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(chunks),
                "chunks_indexed": index_result["chunks_indexed"],
                "processing_time_ms": processing_result["processing_time_ms"],
                "total_time_ms": total_time_ms,
                "metadata": metadata
            }
            
            logger.info(
                "Document ingestion completed",
                document_id=document_id,
                chunks_created=len(chunks),
                total_time_ms=total_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Document ingestion failed",
                error=str(e),
                error_type=type(e).__name__,
                metadata=metadata
            )

            # Re-raise the exception instead of returning error message
            raise Exception(f"Document ingestion failed: {str(e)}")
    
    def _get_cache_key(self, query: str, max_context_chunks: int, confidence_threshold: float) -> str:
        """Generate cache key for query."""
        cache_data = f"{query.lower().strip()}_{max_context_chunks}_{confidence_threshold}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        if cache_key in self._query_cache:
            cached_data = self._query_cache[cache_key]
            # Simple cache expiry - 5 minutes
            if (datetime.utcnow() - cached_data["timestamp"]).seconds < 300:
                logger.info("Cache hit", cache_key=cache_key[:8])
                return cached_data["response"]
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response with timestamp."""
        # Implement simple LRU by removing oldest entries
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = min(self._query_cache.keys(),
                           key=lambda k: self._query_cache[k]["timestamp"])
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = {
            "response": response,
            "timestamp": datetime.utcnow()
        }
        logger.info("Response cached", cache_key=cache_key[:8])

    async def generate_response(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        max_context_chunks: int = 8,
        confidence_threshold: Optional[float] = None,
        channel: str = "default"
    ) -> Dict[str, Any]:
        """
        Generate RAG response for a user query.

        Args:
            query: User query
            conversation_id: Optional conversation identifier
            max_context_chunks: Maximum chunks to use for context
            confidence_threshold: Minimum relevance score for chunks
            channel: Channel type for preprocessing optimization

        Returns:
            Generated response with sources and metadata
        """
        try:
            start_time = datetime.utcnow()
            confidence_threshold = confidence_threshold or self.settings.confidence_threshold

            # Step 1: Process and enhance query if enabled (CONSISTENT CHANNEL)
            if self.settings.enable_query_enhancement:
                enhanced_query_result = await self.query_processor.process_query(
                    query=query,
                    channel="whatsapp"  # Use consistent WhatsApp channel
                )
                processed_query = enhanced_query_result.enhanced_query
                query_metadata = {
                    "original_query": query,
                    "enhanced_query": processed_query,
                    "query_category": enhanced_query_result.query_type.category,
                    "query_intent": enhanced_query_result.query_type.intent,
                    "preprocessing_steps": enhanced_query_result.preprocessing_applied
                }
                logger.info("Query enhanced", **query_metadata)
            else:
                processed_query = query
                query_metadata = {"original_query": query, "enhanced_query": query}

            # Check cache with processed query
            cache_key = self._get_cache_key(processed_query, max_context_chunks, confidence_threshold)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                # Update conversation_id and add query metadata
                cached_response["conversation_id"] = conversation_id
                cached_response.update(query_metadata)
                return cached_response

            # Generate query embedding using processed query
            query_embedding = await self.openai_client.generate_embedding(processed_query)
            
            # Search for relevant chunks
            confidence_threshold = confidence_threshold or self.settings.confidence_threshold
            
            search_results = self.vector_store.search_documents(
                query_vector=query_embedding,
                limit=max_context_chunks,
                score_threshold=confidence_threshold
            )
            
            # Build context from search results
            context_chunks = []
            sources = []

            logger.info("Processing search results", results_count=len(search_results))

            for i, result in enumerate(search_results):
                try:
                    context_chunks.append(result["text"])

                    # Create source document reference with validation
                    source = SourceDocument(
                        document_id=result["document_id"],
                        chunk_id=result["chunk_id"],
                        title=result.get("title"),
                        relevance_score=float(result["score"]),  # Ensure it's a float
                        text_excerpt=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                    )
                    sources.append(source)
                    logger.info(f"Created source {i}", source_type=type(source))

                except Exception as e:
                    logger.error(f"Failed to create source {i}", error=str(e), result_keys=list(result.keys()))
                    continue
            
            # Build context string
            if context_chunks:
                context = "\n\n".join([
                    f"Source {i+1}: {chunk}" 
                    for i, chunk in enumerate(context_chunks)
                ])
            else:
                # When no context is available, provide explicit instructions
                context = "لا توجد معلومات متاحة في قاعدة المعرفة لهذا الاستفسار."
            
            # Detect language and create system prompt for RAG with query context
            detected_language = self._detect_language(query)
            system_prompt = self._build_system_prompt(
                context,
                query_category=query_metadata.get("query_category", "general"),
                language=detected_language,
                has_sources=bool(context_chunks)
            )

            # Generate response using OpenAI with processed query
            messages = [
                {"role": "user", "content": processed_query}
            ]

            chat_result = await self.openai_client.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.0,  # Deterministic responses
                max_tokens=self.settings.max_response_tokens
            )
            
            # Calculate confidence based on source relevance and response quality
            if sources:
                # When sources are available, use relevance scores
                avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
                confidence = min(avg_relevance * 1.2, 1.0)
            else:
                # For general queries without sources, assess response quality
                # Check if the AI generated a meaningful response
                response_length = len(chat_result.get("content", ""))
                if response_length > 50:  # Substantial response
                    confidence = 0.6  # Good confidence for general knowledge
                elif response_length > 20:  # Moderate response
                    confidence = 0.4  # Moderate confidence
                else:
                    confidence = 0.2  # Low confidence for very short responses

            # ESCALATION COMPLETELY REMOVED - No more escalation system
            # This prevents unwanted escalation messages from appearing

            end_time = datetime.utcnow()
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Get raw response content - fail if no content instead of fallback
            if not chat_result["content"]:
                raise Exception("OpenAI returned empty response content")
            raw_response = str(chat_result["content"])

            # Clean response for language consistency
            cleaned_response = self._clean_response_language(raw_response, detected_language)

            # Apply response formatting with WhatsApp-specific formatting
            formatted_response = self.response_formatter.format_response(
                content=cleaned_response,
                category=query_metadata.get("query_category", "general"),
                channel="whatsapp",  # Enable WhatsApp-specific formatting
                confidence=confidence,
                sources_count=len(sources),
                query_intent=query_metadata.get("query_intent", "question")
            )

            final_response = formatted_response.content  # Extract the content string
            
            # Since formatted_response is now a string, we need to handle the missing attributes
            # that the code expects later
            
            # ROOT CAUSE FIXED - No more unwanted text generation
            # The system prompt now prevents escalation messages from being generated

            # Validate and create result with proper types
            result = {
                "response": final_response,
                "confidence": float(confidence),
                "sources": sources,  # List of SourceDocument objects
                "sources_count": int(len(sources)),
                "conversation_id": conversation_id,
                "processing_time_ms": int(total_time_ms),
                "tokens_used": int(chat_result.get("tokens_used", 0)),
                "model_used": str(chat_result.get("model", "unknown")),
                # ESCALATION FIELDS REMOVED - No more escalation system
                "response_tone": formatted_response.tone.value,
                "response_format": formatted_response.format_type.value,
                "channel_optimized": formatted_response.channel_optimized,
                **query_metadata  # Include query processing metadata
            }

            logger.info("RAG result created",
                       result_keys=list(result.keys()),
                       sources_type=type(result["sources"]),
                       sources_count=len(result["sources"]))
            
            logger.info(
                "RAG response generated",
                query_length=len(query),
                sources_count=len(sources),
                confidence=confidence,
                processing_time_ms=total_time_ms,
                tokens_used=chat_result["tokens_used"]
            )

            # Cache the response for future use
            self._cache_response(cache_key, result)

            return result
            
        except Exception as e:
            logger.error(
                "RAG response generation failed",
                query=query[:100],
                conversation_id=conversation_id,
                error=str(e),
                error_type=type(e).__name__
            )

            # Re-raise the exception instead of returning fallback response
            raise Exception(f"RAG response generation failed: {str(e)}")

    def _detect_language(self, text: str) -> str:
        """Detect language from text content."""
        # Count Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])

        if total_chars == 0:
            return "ar"  # Default to Arabic

        arabic_ratio = arabic_chars / total_chars

        # If more than 30% Arabic characters, consider it Arabic
        return "ar" if arabic_ratio > 0.3 else "en"

    def _clean_response_language(self, response: str, target_language: str) -> str:
        """Clean response to ensure language consistency."""
        if target_language == "ar":
            # Remove common English phrases that might be added
            english_phrases = [
                "Is there anything else I can help you with?",
                "Can I help you with anything else?",
                "How can I assist you further?",
                "Let me know if you need any other assistance.",
                "Feel free to ask if you have any other questions."
            ]

            cleaned_response = response
            for phrase in english_phrases:
                cleaned_response = cleaned_response.replace(phrase, "").strip()

            # Remove any remaining English sentences at the end
            lines = cleaned_response.split('\n')
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line:
                    # Check if line is mostly English
                    english_chars = sum(1 for char in line if char.isalpha() and 'a' <= char.lower() <= 'z')
                    total_chars = sum(1 for char in line if char.isalpha())

                    if total_chars > 0:
                        english_ratio = english_chars / total_chars
                        # Skip lines that are mostly English (>70%)
                        if english_ratio <= 0.7:
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)

            return '\n'.join(cleaned_lines).strip()

        return response

    # AGGRESSIVE CLEANING FUNCTION REMOVED - No longer needed
    # Root cause is fixed in the system prompt

    def _build_system_prompt(self, context: str, query_category: str = "general", language: str = "ar", has_sources: bool = True) -> str:
        """
        Build system prompt for RAG response generation.

        Args:
            context: Retrieved context from vector search
            query_category: Category of the query for specialized instructions

        Returns:
            System prompt string
        """
        # Category-specific instructions
        category_instructions = {
            "troubleshooting": """
- Provide clear, step-by-step troubleshooting instructions
- Ask clarifying questions if the issue isn't clear
- Suggest checking common causes first
- If the issue persists, recommend escalation to technical support""",
            "billing": """
- Be precise about billing information and policies
- Explain charges clearly and provide context
- Direct to appropriate billing support if needed
- Handle sensitive financial information with care""",
            "setup": """
- Provide clear, sequential setup instructions
- Break down complex processes into simple steps
- Mention prerequisites and requirements
- Offer to help with next steps after setup""",
            "policies": """
- Quote policies accurately from the provided context
- Explain policy implications clearly
- Direct to full policy documents when appropriate
- Be clear about what is and isn't covered""",
            "general": """
- Provide helpful, comprehensive answers
- Be ready to assist with various types of questions
- Offer additional help or resources when appropriate"""
        }

        specific_instructions = category_instructions.get(query_category, category_instructions["general"])

        if language == "ar":
            if has_sources:
                return f"""أنت SAIA، مساعد ذكي متخصص في خدمة العملاء. مهمتك تقديم إجابات دقيقة ومفيدة باللغة العربية فقط بناءً على السياق المقدم.

⚠️ تعليمات صارمة - لا تخالفها أبداً:
- لا تبدأ إجابتك بـ "أفهم قلقك" أو "أفهم مخاوفك" أو "I understand your concern"
- لا تقل "أريد التأكد من حصولك على أفضل مساعدة ممكنة"
- لا تقترح التواصل مع أخصائي أو دعم بشري
- لا تقدم خيارات أو قوائم إضافية
- ابدأ إجابتك مباشرة بالمعلومات المطلوبة

📱 تنسيق WhatsApp - استخدم هذا التنسيق دائماً:

**1. هيكل واضح:**
- ابدأ بفقرة قصيرة تمهيدية
- ثم اذكر "تشمل الخدمات:" أو "المميزات:"
- استخدم نقاط (•) واحدة فقط لكل عنصر
- انهِ بفقرة ختامية قصيرة

**2. تنسيق النقاط:**
- استخدم (•) واحدة فقط لكل عنصر
- لا تستخدم (• • •) أو نقاط متعددة
- اترك مسافة بين كل نقطة

**3. طول الفقرات:**
- فقرات قصيرة (2-3 جملة كحد أقصى)
- استخدم فواصل واضحة بين الأفكار
- اجعل كل نقطة مفهومة ومحددة

**4. مثال على التنسيق:**
```
وازن هي منصة رقمية متخصصة في التأمين.

تشمل الخدمات:
• تأمين المركبات
• تأمين شامل
• إدارة الوثائق

نقدم تجربة سلسة ومهنية.
```

معلومات السياق:
{context}

فئة الاستفسار: {query_category}

التعليمات المهمة جداً:
1. استخدم المعلومات المتوفرة في السياق أعلاه لتقديم إجابة شاملة ومفيدة
2. إذا وجدت معلومات ذات صلة في السياق، اعتمد عليها بالكامل في إجابتك
3. قدم تفاصيل واضحة ومحددة من المعلومات المتوفرة
4. كن واثقاً في استخدام المعلومات الموجودة في السياق
5. حافظ على نبرة مهذبة ومهنية
6. أجب باللغة العربية فقط - لا تستخدم أي كلمات إنجليزية أبداً
7. لا تقل "المعلومات غير متوفرة" إذا كانت موجودة في السياق
8. كن مساعداً وقدم معلومات قيمة للمستخدم
9. انهِ إجابتك بنقطة (.) ولا تضف أي نص إضافي
10. لا تستخدم أي عبارات إنجليزية مطلقاً
11. استخدم تنسيق WhatsApp المناسب (أرقام، نقاط، فقرات قصيرة)

إرشادات خاصة بالفئة:
{specific_instructions}

مهم جداً: أجب باللغة العربية فقط. لا تضع أي نص إنجليزي في أي جزء من الإجابة.

تذكر: استخدم المعلومات الموجودة في السياق أعلاه لتقديم إجابة شاملة ومفصلة. كن واثقاً في استخدام هذه المعلومات. أجب باللغة العربية فقط."""
            else:
                return f"""أنت SAIA، مساعد ذكي متخصص في خدمة العملاء.

⚠️ تعليمات صارمة - لا تخالفها أبداً:
- لا تقل "أفهم قلقك" أو "أفهم مخاوفك" أو "I understand your concern" 
- لا تقل "أريد التأكد من حصولك على أفضل مساعدة ممكنة"
- لا تقترح التواصل مع أخصائي أو دعم بشري
- لا تقدم خيارات أو قوائم

الوضع الحالي: لا توجد معلومات متاحة في قاعدة المعرفة حول هذا الاستفسار.

تعليماتك:
قل بالضبط: "عذراً، لا أملك معلومات حول هذا الموضوع في قاعدة المعرفة المتاحة لي حالياً. يرجى تجربة سؤال مختلف أو أكثر تحديداً."

لا تضف أي نص آخر."""
        else:
            return f"""You are SAIA, a helpful AI assistant for customer support. Your role is to provide accurate, helpful responses in English only based on the provided context.

⚠️ STRICT INSTRUCTIONS - NEVER VIOLATE THESE:
- NEVER start your response with "I understand your concern" or similar empathetic phrases
- NEVER say "I want to make sure you get the best help possible"
- NEVER suggest contacting a specialist or human support
- NEVER offer additional options or lists
- Start your response DIRECTLY with the requested information

📱 WhatsApp Formatting - Always use this format:
- If the answer contains steps: use numbers (1. 2. 3.)
- If the answer contains points: use bullet points (•)
- If the answer contains options: use bullet points (•)
- Use short, clearly separated paragraphs
- Make the answer easy to read on WhatsApp

CONTEXT INFORMATION:
{context}

QUERY CATEGORY: {query_category.title()}

GENERAL INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your response
4. If you reference specific information, you can mention it comes from the provided sources
5. Maintain a helpful, professional tone
6. If the user asks about something not covered in the context, politely explain that you don't have that information available
7. Use appropriate WhatsApp formatting (numbers, bullet points, short paragraphs)
7. Answer in English only - do not use any Arabic words

CATEGORY-SPECIFIC GUIDANCE:
{specific_instructions}

IMPORTANT: Answer in English only. Do not add any Arabic text in any part of the response.

Remember: Only use the context provided above to answer questions. Do not use external knowledge beyond what's given in the context."""
    
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search documents using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_score: Minimum relevance score
            filters: Optional search filters
            
        Returns:
            Search results with relevance scores
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate query embedding
            query_embedding = await self.openai_client.generate_embedding(query)
            
            # Perform vector search
            search_results = self.vector_store.search_documents(
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
                document_filter=filters
            )
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "chunk_id": result["chunk_id"],
                    "document_id": result["document_id"],
                    "title": result.get("title"),
                    "content": result["text"],
                    "score": result["score"],
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)
            
            result = {
                "results": formatted_results,
                "total_results": len(formatted_results),
                "processing_time_ms": processing_time_ms,
                "query": query
            }
            
            logger.info(
                "Document search completed",
                query=query[:100],
                results_count=len(formatted_results),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Document search failed",
                query=query[:100],
                error=str(e)
            )
            raise


def get_rag_service() -> RAGService:
    """
    Get RAG service instance.
    
    Returns:
        RAG service instance
    """
    return RAGService()


# Export for easy importing
__all__ = ["RAGService", "get_rag_service"]
