"""
SAIA-RAG Fallback Logic Module

Fallback mechanisms for handling cases where RAG cannot provide adequate responses.
Includes escalation logic, confidence assessment, and alternative response strategies.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import structlog

from .config import get_settings

# Get logger
logger = structlog.get_logger()


class FallbackReason(Enum):
    """Reasons for triggering fallback mechanisms."""
    LOW_CONFIDENCE = "low_confidence"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    QUERY_TOO_COMPLEX = "query_too_complex"
    SYSTEM_ERROR = "system_error"
    EXPLICIT_REQUEST = "explicit_request"


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    GENERIC_RESPONSE = "generic_response"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    REQUEST_CLARIFICATION = "request_clarification"
    SUGGEST_ALTERNATIVES = "suggest_alternatives"


class FallbackHandler:
    """
    Handles fallback scenarios when RAG cannot provide adequate responses.
    
    Provides intelligent fallback strategies based on context and user needs.
    """
    
    def __init__(self):
        """Initialize the fallback handler."""
        self.settings = get_settings()
        
        # Fallback response templates
        self.fallback_responses = {
            FallbackReason.LOW_CONFIDENCE: {
                "ar": "عذراً، لا أملك معلومات كافية للإجابة على سؤالك بثقة. هل يمكنك إعادة صياغة السؤال أو تقديم المزيد من التفاصيل؟",
                "en": "I don't have enough information to answer your question confidently. Could you rephrase your question or provide more details?"
            },
            FallbackReason.NO_RELEVANT_CONTEXT: {
                "ar": "لم أجد معلومات ذات صلة بسؤالك في قاعدة المعرفة المتاحة. هل تريد التحدث مع أحد ممثلي خدمة العملاء؟",
                "en": "I couldn't find relevant information for your question in the available knowledge base. Would you like to speak with a customer service representative?"
            },
            FallbackReason.QUERY_TOO_COMPLEX: {
                "ar": "سؤالك معقد ويتطلب مساعدة متخصصة. سأقوم بتحويلك إلى أحد ممثلي خدمة العملاء للحصول على مساعدة أفضل.",
                "en": "Your question is complex and requires specialized assistance. I'll connect you with a customer service representative for better help."
            },
            FallbackReason.SYSTEM_ERROR: {
                "ar": "أعتذر، حدث خطأ تقني. يرجى المحاولة مرة أخرى أو التواصل مع فريق الدعم الفني.",
                "en": "I apologize, a technical error occurred. Please try again or contact technical support."
            }
        }
        
        logger.info("Fallback handler initialized")
    
    def should_trigger_fallback(
        self,
        confidence: float,
        context_chunks: List[Dict[str, Any]],
        query: str
    ) -> tuple[bool, Optional[FallbackReason]]:
        """
        Determine if fallback should be triggered.
        
        Args:
            confidence: Response confidence score
            context_chunks: Retrieved context chunks
            query: User query
            
        Returns:
            Tuple of (should_fallback, reason)
        """
        # Check confidence threshold
        if confidence < self.settings.confidence_threshold:
            return True, FallbackReason.LOW_CONFIDENCE
        
        # Check if no relevant context found
        if not context_chunks:
            return True, FallbackReason.NO_RELEVANT_CONTEXT
        
        # Check for complex queries (heuristic)
        if self._is_complex_query(query):
            return True, FallbackReason.QUERY_TOO_COMPLEX
        
        return False, None
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Heuristic to detect complex queries that might need human assistance.
        
        Args:
            query: User query
            
        Returns:
            True if query appears complex
        """
        # Complex query indicators
        complex_indicators = [
            "legal", "lawsuit", "court", "attorney", "lawyer",
            "emergency", "urgent", "immediate", "asap",
            "complaint", "dispute", "problem", "issue",
            "refund", "cancel", "terminate", "stop"
        ]
        
        query_lower = query.lower()
        
        # Check for multiple questions
        question_count = query_lower.count('?')
        if question_count > 2:
            return True
        
        # Check for complex indicators
        indicator_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        if indicator_count > 1:
            return True
        
        # Check query length (very long queries might be complex)
        if len(query) > 500:
            return True
        
        return False
    
    def generate_fallback_response(
        self,
        reason: FallbackReason,
        query: str,
        language: str = "ar"
    ) -> Dict[str, Any]:
        """
        Generate appropriate fallback response.
        
        Args:
            reason: Reason for fallback
            query: Original user query
            language: Response language (ar/en)
            
        Returns:
            Fallback response data
        """
        # Get base response template
        response_template = self.fallback_responses.get(reason, {})
        base_response = response_template.get(language, response_template.get("en", ""))
        
        # Determine strategy
        strategy = self._determine_strategy(reason)
        
        # Build response
        response_data = {
            "response": base_response,
            "confidence": 0.0,
            "sources": [],
            "fallback_triggered": True,
            "fallback_reason": reason.value,
            "fallback_strategy": strategy.value,
            "requires_escalation": strategy == FallbackStrategy.ESCALATE_TO_HUMAN
        }
        
        # Add strategy-specific enhancements
        if strategy == FallbackStrategy.SUGGEST_ALTERNATIVES:
            response_data["suggestions"] = self._generate_suggestions(query, language)
        elif strategy == FallbackStrategy.REQUEST_CLARIFICATION:
            response_data["clarification_prompts"] = self._generate_clarification_prompts(query, language)
        
        logger.info(
            "Fallback response generated",
            reason=reason.value,
            strategy=strategy.value,
            language=language
        )
        
        return response_data
    
    def _determine_strategy(self, reason: FallbackReason) -> FallbackStrategy:
        """
        Determine appropriate fallback strategy based on reason.
        
        Args:
            reason: Fallback reason
            
        Returns:
            Appropriate fallback strategy
        """
        strategy_mapping = {
            FallbackReason.LOW_CONFIDENCE: FallbackStrategy.REQUEST_CLARIFICATION,
            FallbackReason.NO_RELEVANT_CONTEXT: FallbackStrategy.ESCALATE_TO_HUMAN,
            FallbackReason.QUERY_TOO_COMPLEX: FallbackStrategy.ESCALATE_TO_HUMAN,
            FallbackReason.SYSTEM_ERROR: FallbackStrategy.GENERIC_RESPONSE,
            FallbackReason.EXPLICIT_REQUEST: FallbackStrategy.ESCALATE_TO_HUMAN
        }
        
        return strategy_mapping.get(reason, FallbackStrategy.GENERIC_RESPONSE)
    
    def _generate_suggestions(self, query: str, language: str) -> List[str]:
        """
        Generate alternative suggestions for the user.
        
        Args:
            query: Original query
            language: Response language
            
        Returns:
            List of suggestions
        """
        if language == "ar":
            return [
                "جرب إعادة صياغة السؤال بكلمات مختلفة",
                "تأكد من أن السؤال واضح ومحدد",
                "اطلب المساعدة من ممثل خدمة العملاء"
            ]
        else:
            return [
                "Try rephrasing your question with different words",
                "Make sure your question is clear and specific",
                "Ask for help from a customer service representative"
            ]
    
    def _generate_clarification_prompts(self, query: str, language: str) -> List[str]:
        """
        Generate clarification prompts for the user.
        
        Args:
            query: Original query
            language: Response language
            
        Returns:
            List of clarification prompts
        """
        if language == "ar":
            return [
                "هل يمكنك تقديم المزيد من التفاصيل؟",
                "ما هو السياق المحدد لسؤالك؟",
                "هل تبحث عن معلومات حول موضوع معين؟"
            ]
        else:
            return [
                "Could you provide more details?",
                "What is the specific context of your question?",
                "Are you looking for information about a particular topic?"
            ]


# Global instance
_fallback_handler = None


def get_fallback_handler() -> FallbackHandler:
    """
    Get global fallback handler instance.
    
    Returns:
        FallbackHandler instance
    """
    global _fallback_handler
    if _fallback_handler is None:
        _fallback_handler = FallbackHandler()
    return _fallback_handler
