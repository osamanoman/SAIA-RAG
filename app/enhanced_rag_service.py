"""
SAIA-RAG Enhanced RAG Service

Integrates all advanced AI capabilities including adaptive retrieval,
intelligent reranking, and conversation memory for optimal customer support.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .openai_client import get_openai_client
from .adaptive_retriever import get_adaptive_retriever, RetrievalStrategy
from .intelligent_reranker import get_intelligent_reranker, RerankingMethod
from .conversation_memory import get_conversation_memory_manager, MessageType, UserSentiment
from .query_processor import get_query_processor

logger = structlog.get_logger()


class ResponseQuality(str, Enum):
    """Response quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class EnhancedChatRequest(BaseModel):
    """Enhanced chat request with conversation context."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    language_preference: str = Field(default="ar", description="User language preference")
    context_preferences: Dict[str, Any] = Field(default_factory=dict, description="User context preferences")


class EnhancedChatResponse(BaseModel):
    """Enhanced chat response with comprehensive metadata."""
    message: str = Field(..., description="AI response message")
    conversation_id: str = Field(..., description="Conversation identifier")
    confidence_score: float = Field(..., description="Response confidence score")
    response_quality: ResponseQuality = Field(..., description="Response quality assessment")
    
    # Source information
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Knowledge sources used")
    source_categories: List[str] = Field(default_factory=list, description="Source categories")
    
    # Processing metadata
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Retrieval processing info")
    reranking_metadata: Dict[str, Any] = Field(default_factory=dict, description="Reranking processing info")
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation context info")
    
    # Recommendations
    escalation_recommended: bool = Field(default=False, description="Whether escalation is recommended")
    follow_up_suggestions: List[str] = Field(default_factory=list, description="Follow-up question suggestions")
    
    # Performance metrics
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")


class EnhancedRAGService:
    """
    Enhanced RAG service integrating all advanced AI capabilities.
    
    Features:
    - Adaptive retrieval strategy selection
    - Intelligent reranking with customer support optimization
    - Conversation memory and context management
    - Multi-turn conversation support
    - Escalation detection and recommendations
    - Response quality assessment
    """
    
    def __init__(self):
        """Initialize enhanced RAG service."""
        self.settings = get_settings()
        self.openai_client = get_openai_client()
        self.adaptive_retriever = get_adaptive_retriever()
        self.intelligent_reranker = get_intelligent_reranker()
        self.conversation_manager = get_conversation_memory_manager()
        self.query_processor = get_query_processor()
        
        # Response quality thresholds
        self.quality_thresholds = {
            ResponseQuality.EXCELLENT: 0.8,
            ResponseQuality.GOOD: 0.6,
            ResponseQuality.ACCEPTABLE: 0.4,
            ResponseQuality.POOR: 0.0
        }
        
        logger.info("Enhanced RAG service initialized")
    
    async def process_chat_request(self, request: EnhancedChatRequest) -> EnhancedChatResponse:
        """
        Process enhanced chat request with full AI capabilities.
        
        Args:
            request: Enhanced chat request
            
        Returns:
            Enhanced chat response with comprehensive metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Initialize or retrieve conversation
            conversation = await self._initialize_conversation(request)
            
            # Step 2: Add user message to conversation
            await self.conversation_manager.add_message(
                conversation_id=conversation.conversation_id,
                content=request.message,
                message_type=MessageType.USER_QUERY
            )
            
            # Step 3: Get conversation context
            conversation_context = await self.conversation_manager.get_conversation_context(
                conversation_id=conversation.conversation_id,
                include_messages=True
            )
            
            # Step 4: Process query with enhanced understanding
            enhanced_query = await self.query_processor.process_query(request.message)
            
            # Step 5: Adaptive retrieval
            retrieval_result = await self.adaptive_retriever.retrieve_adaptive(
                query=request.message,
                conversation_context=conversation_context,
                user_context={
                    "language_preference": request.language_preference,
                    "preferences": request.context_preferences
                }
            )
            
            # Step 6: Intelligent reranking
            reranking_method = self._select_reranking_method(enhanced_query, conversation_context)
            reranking_result = await self.intelligent_reranker.rerank_results(
                query=request.message,
                search_results=retrieval_result.chunks,
                method=reranking_method,
                conversation_context=conversation_context
            )
            
            # Step 7: Detect language and generate AI response
            detected_language = self._detect_language(request.message)
            ai_response = await self._generate_ai_response(
                query=request.message,
                context_chunks=reranking_result.reranked_chunks,
                conversation_context=conversation_context,
                language_preference=detected_language
            )
            
            # Step 8: Assess response quality
            response_quality = self._assess_response_quality(
                ai_response, reranking_result.reranked_scores
            )
            
            # Step 9: Check escalation triggers
            escalation_recommended = await self.conversation_manager.check_escalation_triggers(
                conversation_id=conversation.conversation_id,
                message_content=request.message,
                confidence_score=ai_response.get("confidence_score", 0.0)
            )
            
            # Step 10: Add AI response to conversation
            await self.conversation_manager.add_message(
                conversation_id=conversation.conversation_id,
                content=ai_response["content"],
                message_type=MessageType.AI_RESPONSE,
                confidence_score=ai_response.get("confidence_score", 0.0),
                sources=[chunk.get("payload", {}).get("source", "") for chunk in reranking_result.reranked_chunks]
            )
            
            # Step 11: Generate follow-up suggestions
            follow_up_suggestions = await self._generate_follow_up_suggestions(
                conversation_context, enhanced_query
            )
            
            # Step 12: Compile response
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            response = EnhancedChatResponse(
                message=ai_response["content"],
                conversation_id=conversation.conversation_id,
                confidence_score=ai_response.get("confidence_score", 0.0),
                response_quality=response_quality,
                sources=self._format_sources(reranking_result.reranked_chunks),
                source_categories=list(set(
                    chunk.get("payload", {}).get("category", "unknown") 
                    for chunk in reranking_result.reranked_chunks
                )),
                retrieval_metadata={
                    "strategy_used": retrieval_result.strategy_used.value,
                    "total_candidates": retrieval_result.total_candidates,
                    "chunks_retrieved": len(retrieval_result.chunks),
                    "category_distribution": retrieval_result.category_distribution
                },
                reranking_metadata={
                    "method_used": reranking_result.method_used.value,
                    "score_improvement": reranking_result.processing_metadata.get("avg_score_improvement", 0.0),
                    "diversification_applied": reranking_result.processing_metadata.get("diversification_applied", False)
                },
                conversation_metadata={
                    "total_messages": conversation_context.get("total_messages", 0),
                    "user_sentiment": conversation_context.get("user_sentiment", "neutral"),
                    "topics": conversation_context.get("topics", []),
                    "session_duration_minutes": conversation_context.get("session_duration_minutes", 0)
                },
                escalation_recommended=escalation_recommended,
                follow_up_suggestions=follow_up_suggestions,
                processing_time_ms=processing_time
            )
            
            logger.info(
                "Enhanced chat request processed",
                conversation_id=conversation.conversation_id,
                response_quality=response_quality.value,
                confidence_score=ai_response.get("confidence_score", 0.0),
                escalation_recommended=escalation_recommended,
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error("Enhanced chat request failed", error=str(e))
            # Return fallback response
            return await self._create_fallback_response(request, str(e))
    
    async def _initialize_conversation(self, request: EnhancedChatRequest):
        """Initialize or retrieve conversation."""
        if request.conversation_id:
            # Try to get existing conversation
            conversation = await self.conversation_manager.get_conversation(request.conversation_id)
            if conversation:
                return conversation
        
        # Start new conversation
        return await self.conversation_manager.start_conversation(
            user_id=request.user_id,
            session_id=request.session_id,
            initial_context={
                "language": request.language_preference,
                "patterns": request.context_preferences
            }
        )
    
    def _select_reranking_method(
        self,
        enhanced_query: Any,
        conversation_context: Dict[str, Any]
    ) -> RerankingMethod:
        """Select optimal reranking method based on query and context."""
        
        # Check user sentiment
        user_sentiment = conversation_context.get("user_sentiment", "neutral")
        if user_sentiment in ["frustrated", "angry"]:
            return RerankingMethod.CUSTOMER_SUPPORT
        
        # Check query complexity
        query_category = enhanced_query.query_type.category
        if query_category in ["troubleshooting", "complex"]:
            return RerankingMethod.HYBRID
        
        # Check conversation length
        total_messages = conversation_context.get("total_messages", 0)
        if total_messages > 10:
            return RerankingMethod.CONTEXTUAL
        
        # Default to customer support method
        return RerankingMethod.CUSTOMER_SUPPORT
    
    async def _generate_ai_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_context: Dict[str, Any],
        language_preference: str = "ar"
    ) -> Dict[str, Any]:
        """Generate AI response using context chunks and conversation history."""
        
        # Prepare context for AI
        context_text = "\n\n".join([
            chunk.get("payload", {}).get("text", "") 
            for chunk in context_chunks[:5]  # Use top 5 chunks
        ])
        
        # Include conversation history
        recent_messages = conversation_context.get("recent_messages", [])
        conversation_history = "\n".join([
            f"{msg['type']}: {msg['content']}" 
            for msg in recent_messages[-3:]  # Last 3 messages
        ])
        
        # Prepare system prompt
        system_prompt = self._create_system_prompt(language_preference, conversation_context)
        
        # Prepare user prompt based on language
        if language_preference == "ar":
            user_prompt = f"""
السياق المتاح:
{context_text}

تاريخ المحادثة:
{conversation_history}

السؤال الحالي: {query}

يرجى تقديم إجابة مفيدة ودقيقة باللغة العربية فقط. لا تستخدم أي كلمات إنجليزية.
"""
        else:
            user_prompt = f"""
Available Context:
{context_text}

Conversation History:
{conversation_history}

Current Question: {query}

Please provide a helpful and accurate answer in English only. Do not use any Arabic words.
"""
        
        try:
            # Generate response using OpenAI
            response = await self.openai_client.generate_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Calculate confidence score based on context quality
            confidence_score = self._calculate_confidence_score(context_chunks, query)

            # Clean response for language consistency
            raw_content = response.choices[0].message.content
            cleaned_content = self._clean_response_language(raw_content, language_preference)

            return {
                "content": cleaned_content,
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.error("AI response generation failed", error=str(e))
            return {
                "content": "أعتذر، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى.",
                "confidence_score": 0.1
            }
    
    def _create_system_prompt(
        self,
        language_preference: str,
        conversation_context: Dict[str, Any]
    ) -> str:
        """Create system prompt based on context."""

        if language_preference == "ar":
            base_prompt = """أنت مساعد ذكي متخصص في خدمة العملاء للتأمين.
مهمتك هي تقديم إجابات دقيقة ومفيدة باللغة العربية فقط.

إرشادات مهمة:
- أجب باللغة العربية فقط - لا تستخدم أي كلمات إنجليزية
- استخدم المعلومات المتوفرة في السياق فقط
- كن مهذباً ومفيداً
- إذا لم تجد إجابة في السياق، اعتذر واقترح التواصل مع خدمة العملاء
- اجعل إجاباتك واضحة ومباشرة
- لا تضع أي نص إنجليزي في نهاية الإجابة"""
        else:
            base_prompt = """You are an intelligent customer service assistant specialized in insurance.
Your task is to provide accurate and helpful answers in English only.

Important guidelines:
- Answer in English only - do not use any Arabic words
- Use only the information available in the context
- Be polite and helpful
- If you cannot find an answer in the context, apologize and suggest contacting customer service
- Make your answers clear and direct
- Do not add any Arabic text at the end of your response"""

        # Add context-specific instructions
        user_sentiment = conversation_context.get("user_sentiment", "neutral")
        if language_preference == "ar":
            if user_sentiment == "frustrated":
                base_prompt += "\n- العميل يبدو محبطاً، كن أكثر تفهماً وقدم حلول سريعة"
            elif user_sentiment == "confused":
                base_prompt += "\n- العميل يبدو محتاراً، قدم شرح مفصل وواضح"
        else:
            if user_sentiment == "frustrated":
                base_prompt += "\n- The customer seems frustrated, be more understanding and provide quick solutions"
            elif user_sentiment == "confused":
                base_prompt += "\n- The customer seems confused, provide detailed and clear explanations"

        return base_prompt

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

    def _calculate_confidence_score(
        self,
        context_chunks: List[Dict[str, Any]],
        query: str
    ) -> float:
        """Calculate confidence score based on context quality."""

        if not context_chunks:
            return 0.1

        # Base confidence from chunk scores
        chunk_scores = [chunk.get("score", 0.0) for chunk in context_chunks]
        avg_chunk_score = sum(chunk_scores) / len(chunk_scores)

        # Adjust based on number of relevant chunks
        chunk_count_factor = min(1.0, len(context_chunks) / 3)

        # Adjust based on query-context alignment
        query_words = set(query.lower().split())
        context_text = " ".join([
            chunk.get("payload", {}).get("text", "")
            for chunk in context_chunks[:3]
        ]).lower()
        context_words = set(context_text.split())

        word_overlap = len(query_words.intersection(context_words))
        overlap_factor = min(1.0, word_overlap / max(1, len(query_words)))

        # Calculate final confidence
        confidence = (avg_chunk_score * 0.5) + (chunk_count_factor * 0.3) + (overlap_factor * 0.2)

        return min(0.95, max(0.1, confidence))

    def _assess_response_quality(
        self,
        ai_response: Dict[str, Any],
        reranked_scores: List[float]
    ) -> ResponseQuality:
        """Assess response quality based on multiple factors."""

        confidence_score = ai_response.get("confidence_score", 0.0)

        # Factor in reranking scores
        avg_rerank_score = sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0.0

        # Combined quality score
        quality_score = (confidence_score * 0.7) + (avg_rerank_score * 0.3)

        # Determine quality level
        if quality_score >= self.quality_thresholds[ResponseQuality.EXCELLENT]:
            return ResponseQuality.EXCELLENT
        elif quality_score >= self.quality_thresholds[ResponseQuality.GOOD]:
            return ResponseQuality.GOOD
        elif quality_score >= self.quality_thresholds[ResponseQuality.ACCEPTABLE]:
            return ResponseQuality.ACCEPTABLE
        else:
            return ResponseQuality.POOR

    async def _generate_follow_up_suggestions(
        self,
        conversation_context: Dict[str, Any],
        enhanced_query: Any
    ) -> List[str]:
        """Generate follow-up question suggestions."""

        suggestions = []

        # Based on topics discussed
        topics = conversation_context.get("topics", [])

        if "insurance" in topics:
            suggestions.extend([
                "كيف يمكنني تجديد وثيقة التأمين؟",
                "ما هي التغطيات المتاحة؟"
            ])

        if "payment" in topics:
            suggestions.extend([
                "كيف يمكنني دفع القسط؟",
                "ما هي طرق الدفع المتاحة؟"
            ])

        if "claim" in topics:
            suggestions.extend([
                "كيف أقدم مطالبة تأمين؟",
                "ما هي المستندات المطلوبة؟"
            ])

        # Based on query category
        query_category = enhanced_query.query_type.category

        if query_category == "faq":
            suggestions.append("هل لديك أسئلة أخرى؟")
        elif query_category == "troubleshooting":
            suggestions.append("هل تم حل المشكلة؟")

        # Limit to 3 suggestions
        return suggestions[:3]

    def _format_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for response."""

        sources = []

        for i, chunk in enumerate(context_chunks[:5]):  # Top 5 sources
            payload = chunk.get("payload", {})

            source = {
                "id": f"source_{i+1}",
                "title": payload.get("title", f"مصدر {i+1}"),
                "category": payload.get("category", "unknown"),
                "relevance_score": chunk.get("score", 0.0),
                "text_preview": payload.get("text", "")[:200] + "..." if len(payload.get("text", "")) > 200 else payload.get("text", "")
            }

            sources.append(source)

        return sources

    async def _create_fallback_response(
        self,
        request: EnhancedChatRequest,
        error_message: str
    ) -> EnhancedChatResponse:
        """Create fallback response when processing fails."""

        fallback_message = "أعتذر، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى أو التواصل مع خدمة العملاء."

        return EnhancedChatResponse(
            message=fallback_message,
            conversation_id=request.conversation_id or "fallback",
            confidence_score=0.1,
            response_quality=ResponseQuality.POOR,
            sources=[],
            source_categories=[],
            retrieval_metadata={"error": "retrieval_failed"},
            reranking_metadata={"error": "reranking_failed"},
            conversation_metadata={"error": "conversation_failed"},
            escalation_recommended=True,
            follow_up_suggestions=["التواصل مع خدمة العملاء"],
            processing_time_ms=0
        )

    async def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation summary and analytics."""

        conversation = await self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            return None

        context = await self.conversation_manager.get_conversation_context(
            conversation_id, include_messages=True
        )

        return {
            "conversation_id": conversation_id,
            "state": conversation.state.value,
            "total_messages": conversation.total_messages,
            "user_sentiment": conversation.user_sentiment.value,
            "topics_discussed": conversation.topics,
            "categories_accessed": conversation.categories,
            "resolution_attempts": conversation.resolution_attempts,
            "escalation_triggers": conversation.escalation_triggers,
            "session_duration_minutes": context.get("session_duration_minutes", 0),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat()
        }

    async def cleanup_resources(self) -> Dict[str, int]:
        """Cleanup resources and return statistics."""

        # Cleanup idle conversations
        cleaned_conversations = await self.conversation_manager.cleanup_idle_conversations()

        # Get analytics
        analytics = await self.conversation_manager.get_conversation_analytics()

        return {
            "cleaned_conversations": cleaned_conversations,
            "active_conversations": analytics.get("total_conversations", 0),
            "total_messages": analytics.get("total_messages", 0)
        }


# Global instance
_enhanced_rag_service = None


def get_enhanced_rag_service() -> EnhancedRAGService:
    """Get global enhanced RAG service instance."""
    global _enhanced_rag_service
    if _enhanced_rag_service is None:
        _enhanced_rag_service = EnhancedRAGService()
    return _enhanced_rag_service
