"""
SAIA-RAG Escalation Module

Implements smart escalation logic for customer support scenarios.
Handles low confidence responses and provides human handoff options.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .config import get_settings

logger = structlog.get_logger()


class EscalationType(str, Enum):
    """Types of escalation scenarios."""
    LOW_CONFIDENCE = "low_confidence"
    NO_RESULTS = "no_results"
    COMPLEX_QUERY = "complex_query"
    TECHNICAL_ISSUE = "technical_issue"
    BILLING_ISSUE = "billing_issue"
    COMPLAINT = "complaint"
    URGENT = "urgent"


class EscalationReason(BaseModel):
    """Escalation reason details."""
    type: EscalationType = Field(..., description="Type of escalation")
    confidence: float = Field(..., description="Response confidence score")
    threshold: float = Field(..., description="Confidence threshold used")
    category: str = Field(..., description="Query category")
    description: str = Field(..., description="Human-readable escalation reason")
    suggested_action: str = Field(..., description="Suggested next action")


class EscalationResponse(BaseModel):
    """Escalation response with options."""
    should_escalate: bool = Field(..., description="Whether to escalate")
    reason: Optional[EscalationReason] = Field(None, description="Escalation reason if applicable")
    escalation_message: str = Field(..., description="Message to show user")
    fallback_response: str = Field(..., description="Fallback response if no escalation")
    escalation_options: List[str] = Field(default_factory=list, description="Available escalation options")


class EscalationManager:
    """
    Manages escalation logic for customer support scenarios.
    
    Determines when and how to escalate based on:
    - Response confidence
    - Query category
    - Channel type
    - Previous escalations
    """
    
    def __init__(self):
        """Initialize escalation manager."""
        self.settings = get_settings()
        
        # Category-specific escalation thresholds
        self.category_thresholds = {
            "troubleshooting": 0.35,  # Technical issues need higher confidence
            "billing": 0.45,          # Billing requires high accuracy
            "setup": 0.30,            # Setup can be more flexible
            "policies": 0.40,         # Policies need accuracy
            "general": 0.35           # General queries
        }
        
        # Escalation messages by category
        self.escalation_messages = {
            "troubleshooting": "I want to make sure you get the most accurate technical assistance. Let me connect you with a technical specialist who can help resolve this issue.",
            "billing": "For billing and payment questions, I'd like to connect you with our billing specialist who can provide detailed assistance with your account.",
            "setup": "To ensure your setup goes smoothly, let me connect you with a setup specialist who can guide you through the process step by step.",
            "policies": "For detailed policy information, I'd like to connect you with a specialist who can provide comprehensive guidance on our terms and policies.",
            "general": "I want to make sure you get the best possible assistance. Let me connect you with a specialist who can help you further."
        }
        
        logger.info("Escalation manager initialized")
    
    def should_escalate(
        self,
        confidence: float,
        category: str = "general",
        channel: str = "default",
        query_intent: str = "question",
        sources_count: int = 0
    ) -> EscalationResponse:
        """
        Determine if a response should be escalated.
        
        Args:
            confidence: Response confidence score
            category: Query category
            channel: Communication channel
            query_intent: User intent (question, complaint, etc.)
            sources_count: Number of sources found
            
        Returns:
            Escalation decision with details
        """
        try:
            # Get category-specific threshold
            threshold = self.category_thresholds.get(category, self.settings.escalation_threshold)
            
            # Adjust threshold based on channel
            if channel.lower() == "whatsapp":
                threshold *= 0.9  # Slightly lower threshold for WhatsApp
            
            # Check escalation conditions
            escalation_reason = self._evaluate_escalation_conditions(
                confidence, threshold, category, query_intent, sources_count
            )
            
            if escalation_reason:
                return self._create_escalation_response(escalation_reason, channel)
            else:
                return EscalationResponse(
                    should_escalate=False,
                    escalation_message="",
                    fallback_response="",
                    escalation_options=[]
                )
                
        except Exception as e:
            logger.error("Escalation evaluation failed", error=str(e))
            # Safe fallback - don't escalate on error
            return EscalationResponse(
                should_escalate=False,
                escalation_message="",
                fallback_response="",
                escalation_options=[]
            )
    
    def _evaluate_escalation_conditions(
        self,
        confidence: float,
        threshold: float,
        category: str,
        query_intent: str,
        sources_count: int
    ) -> Optional[EscalationReason]:
        """Evaluate conditions that might trigger escalation."""
        
        # Primary condition: Low confidence
        if confidence < threshold:
            escalation_type = EscalationType.LOW_CONFIDENCE
            description = f"Response confidence ({confidence:.2f}) below threshold ({threshold:.2f})"
            suggested_action = "Connect with specialist for accurate assistance"
            
            return EscalationReason(
                type=escalation_type,
                confidence=confidence,
                threshold=threshold,
                category=category,
                description=description,
                suggested_action=suggested_action
            )
        
        # No relevant sources found
        if sources_count == 0:
            return EscalationReason(
                type=EscalationType.NO_RESULTS,
                confidence=confidence,
                threshold=threshold,
                category=category,
                description="No relevant information found in knowledge base",
                suggested_action="Connect with specialist for comprehensive assistance"
            )
        
        # Intent-based escalation
        if query_intent == "complaint":
            return EscalationReason(
                type=EscalationType.COMPLAINT,
                confidence=confidence,
                threshold=threshold,
                category=category,
                description="Customer complaint detected",
                suggested_action="Connect with customer service specialist"
            )
        
        # Category-specific escalation for critical areas
        if category == "billing" and confidence < 0.6:
            return EscalationReason(
                type=EscalationType.BILLING_ISSUE,
                confidence=confidence,
                threshold=0.6,
                category=category,
                description="Billing query requires high confidence",
                suggested_action="Connect with billing specialist"
            )
        
        return None
    
    def _create_escalation_response(
        self,
        reason: EscalationReason,
        channel: str
    ) -> EscalationResponse:
        """Create escalation response with appropriate options."""
        
        # Get category-specific message
        base_message = self.escalation_messages.get(
            reason.category,
            self.escalation_messages["general"]
        )
        
        # Channel-specific escalation options
        if channel.lower() == "whatsapp":
            escalation_options = [
                "Connect me with a specialist",
                "I'll try a different question",
                "Send me more information"
            ]
            escalation_message = f"{base_message}\n\nWould you like me to connect you with a specialist?"
        else:
            escalation_options = [
                "Connect with specialist",
                "Try different approach",
                "Get more information",
                "Contact support directly"
            ]
            escalation_message = base_message
        
        # Fallback response for when escalation is declined
        fallback_response = self._generate_fallback_response(reason.category, channel)
        
        return EscalationResponse(
            should_escalate=True,
            reason=reason,
            escalation_message=escalation_message,
            fallback_response=fallback_response,
            escalation_options=escalation_options
        )
    
    def _generate_fallback_response(self, category: str, channel: str) -> str:
        """Generate fallback response when escalation is declined."""
        
        fallback_messages = {
            "troubleshooting": "I'll do my best to help. Could you provide more specific details about the issue you're experiencing?",
            "billing": "I understand you have billing questions. Could you tell me more specifically what you'd like to know?",
            "setup": "I'm here to help with setup. Could you let me know which specific step you're having trouble with?",
            "policies": "I can help with policy questions. Could you be more specific about what policy information you need?",
            "general": "I'm here to help. Could you provide more details about what you're looking for?"
        }
        
        base_message = fallback_messages.get(category, fallback_messages["general"])
        
        if channel.lower() == "whatsapp":
            return f"{base_message} Feel free to ask me anything else!"
        else:
            return f"{base_message} I'm here to assist you with any questions you might have."
    
    def format_escalation_for_channel(
        self,
        escalation: EscalationResponse,
        channel: str,
        include_options: bool = True
    ) -> str:
        """Format escalation message for specific channel."""
        
        if not escalation.should_escalate:
            return ""
        
        message = escalation.escalation_message
        
        if include_options and escalation.escalation_options:
            if channel.lower() == "whatsapp":
                # Simple format for WhatsApp
                return message
            else:
                # More detailed format for web chat
                options_text = "\n".join([f"• {option}" for option in escalation.escalation_options])
                message += f"\n\nOptions:\n{options_text}"
        
        return message
    
    def log_escalation(
        self,
        escalation: EscalationResponse,
        query: str,
        conversation_id: Optional[str] = None
    ):
        """Log escalation for analytics and improvement."""
        
        if escalation.should_escalate and escalation.reason:
            logger.info(
                "Escalation triggered",
                escalation_type=escalation.reason.type.value,
                confidence=escalation.reason.confidence,
                threshold=escalation.reason.threshold,
                category=escalation.reason.category,
                query_preview=query[:100],
                conversation_id=conversation_id
            )


# Global instance
_escalation_manager = None


def get_escalation_manager() -> EscalationManager:
    """Get global escalation manager instance."""
    global _escalation_manager
    if _escalation_manager is None:
        _escalation_manager = EscalationManager()
    return _escalation_manager
