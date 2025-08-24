"""
SAIA-RAG Response Formatting Module

Provides customer support specific response formatting with proper tone,
structure, and helpful formatting for different channels and scenarios.
"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .config import get_settings

logger = structlog.get_logger()


class ResponseTone(str, Enum):
    """Response tone options."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"
    TECHNICAL = "technical"
    URGENT = "urgent"


class ResponseFormat(str, Enum):
    """Response format options."""
    PLAIN = "plain"
    STRUCTURED = "structured"
    STEP_BY_STEP = "step_by_step"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"


class FormattedResponse(BaseModel):
    """Formatted response with metadata."""
    content: str = Field(..., description="Formatted response content")
    tone: ResponseTone = Field(..., description="Applied tone")
    format_type: ResponseFormat = Field(..., description="Applied format")
    channel_optimized: bool = Field(..., description="Whether optimized for specific channel")
    formatting_applied: List[str] = Field(default_factory=list, description="List of formatting steps applied")


class ResponseFormatter:
    """
    Formats responses for optimal customer support experience.
    
    Handles:
    - Channel-specific formatting (WhatsApp, web chat, email)
    - Category-specific structure (troubleshooting, billing, setup)
    - Tone adjustment based on context
    - Consistent formatting and structure
    """
    
    def __init__(self):
        """Initialize response formatter."""
        self.settings = get_settings()
        
        # Category-specific formatting patterns
        self.category_formats = {
            "troubleshooting": ResponseFormat.TROUBLESHOOTING,
            "billing": ResponseFormat.STRUCTURED,
            "setup": ResponseFormat.STEP_BY_STEP,
            "policies": ResponseFormat.STRUCTURED,
            "general": ResponseFormat.PLAIN
        }
        
        # Category-specific tones
        self.category_tones = {
            "troubleshooting": ResponseTone.TECHNICAL,
            "billing": ResponseTone.PROFESSIONAL,
            "setup": ResponseTone.FRIENDLY,
            "policies": ResponseTone.PROFESSIONAL,
            "general": ResponseTone.FRIENDLY
        }
        
        logger.info("Response formatter initialized")
    
    def format_response(
        self,
        content: str,
        category: str = "general",
        channel: str = "default",
        confidence: float = 1.0,
        sources_count: int = 0,
        query_intent: str = "question"
    ) -> FormattedResponse:
        """
        Format response for optimal customer experience.
        
        Args:
            content: Raw response content
            category: Query category
            channel: Communication channel
            confidence: Response confidence
            sources_count: Number of sources used
            query_intent: User intent
            
        Returns:
            Formatted response with metadata
        """
        try:
            formatting_steps = []
            
            # Step 1: Determine tone and format
            tone = self._determine_tone(category, query_intent, confidence)
            format_type = self._determine_format(category, content)
            
            # Step 2: Apply basic formatting
            formatted_content = self._apply_basic_formatting(content)
            formatting_steps.append("basic_formatting")
            
            # Step 3: Apply category-specific formatting
            formatted_content = self._apply_category_formatting(
                formatted_content, category, format_type
            )
            formatting_steps.append("category_formatting")
            
            # Step 4: Apply tone adjustments
            formatted_content = self._apply_tone_formatting(
                formatted_content, tone, query_intent
            )
            formatting_steps.append("tone_formatting")
            
            # Step 5: Apply channel-specific optimizations
            formatted_content, channel_optimized = self._apply_channel_formatting(
                formatted_content, channel
            )
            if channel_optimized:
                formatting_steps.append("channel_optimization")
            
            # Step 6: Add helpful elements
            formatted_content = self._add_helpful_elements(
                formatted_content, category, confidence, sources_count
            )
            formatting_steps.append("helpful_elements")
            
            return FormattedResponse(
                content=formatted_content,
                tone=tone,
                format_type=format_type,
                channel_optimized=channel_optimized,
                formatting_applied=formatting_steps
            )
            
        except Exception as e:
            logger.error("Response formatting failed", error=str(e))
            # Return minimally formatted response on error
            return FormattedResponse(
                content=content,
                tone=ResponseTone.PROFESSIONAL,
                format_type=ResponseFormat.PLAIN,
                channel_optimized=False,
                formatting_applied=["error_fallback"]
            )
    
    def _determine_tone(self, category: str, query_intent: str, confidence: float) -> ResponseTone:
        """Determine appropriate tone based on context."""
        
        # Intent-based tone adjustment
        if query_intent == "complaint":
            return ResponseTone.EMPATHETIC
        elif query_intent == "urgent":
            return ResponseTone.URGENT
        elif confidence < 0.3:
            return ResponseTone.EMPATHETIC
        
        # Category-based tone
        return self.category_tones.get(category, ResponseTone.FRIENDLY)
    
    def _determine_format(self, category: str, content: str) -> ResponseFormat:
        """Determine appropriate format based on category and content."""
        
        # Check if content suggests step-by-step format
        if any(indicator in content.lower() for indicator in ["step", "first", "then", "next", "finally"]):
            return ResponseFormat.STEP_BY_STEP
        
        # Check if content suggests FAQ format
        if content.count("?") > 2 or "question" in content.lower():
            return ResponseFormat.FAQ
        
        # Use category default
        return self.category_formats.get(category, ResponseFormat.PLAIN)
    
    def _apply_basic_formatting(self, content: str) -> str:
        """Apply basic formatting improvements."""
        
        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Ensure proper sentence endings
        if content and not content.endswith(('.', '!', '?')):
            content += '.'
        
        # Fix common formatting issues
        content = re.sub(r'\s+([,.!?])', r'\1', content)  # Remove space before punctuation
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)  # Ensure space after sentence endings
        
        return content
    
    def _apply_category_formatting(self, content: str, category: str, format_type: ResponseFormat) -> str:
        """Apply category-specific formatting."""
        
        if format_type == ResponseFormat.STEP_BY_STEP:
            return self._format_as_steps(content)
        elif format_type == ResponseFormat.TROUBLESHOOTING:
            return self._format_as_troubleshooting(content)
        elif format_type == ResponseFormat.STRUCTURED:
            return self._format_as_structured(content)
        
        return content
    
    def _format_as_steps(self, content: str) -> str:
        """Format content as step-by-step instructions."""
        
        # Look for step indicators and format them
        step_patterns = [
            r'(first|1\.?\s*)',
            r'(then|next|2\.?\s*)',
            r'(after that|3\.?\s*)',
            r'(finally|last|4\.?\s*)'
        ]
        
        formatted = content
        for i, pattern in enumerate(step_patterns, 1):
            formatted = re.sub(
                pattern, 
                f'\n\n**Step {i}:** ', 
                formatted, 
                flags=re.IGNORECASE
            )
        
        return formatted.strip()
    
    def _format_as_troubleshooting(self, content: str) -> str:
        """Format content as troubleshooting guide."""
        
        # Add troubleshooting structure
        if "try" in content.lower() or "check" in content.lower():
            formatted = f"**Troubleshooting Steps:**\n\n{content}"
            
            # Format common troubleshooting actions
            formatted = re.sub(
                r'(try|check|verify|ensure)\s+',
                r'• \1 ',
                formatted,
                flags=re.IGNORECASE
            )
            
            return formatted
        
        return content
    
    def _format_as_structured(self, content: str) -> str:
        """Format content with clear structure."""
        
        # Add structure for policy or billing information
        sentences = content.split('. ')
        if len(sentences) > 2:
            # Group related sentences
            formatted = f"**Information:**\n\n{sentences[0]}.\n\n"
            if len(sentences) > 1:
                formatted += f"**Details:**\n\n{'. '.join(sentences[1:])}."
            return formatted
        
        return content
    
    def _apply_tone_formatting(self, content: str, tone: ResponseTone, query_intent: str) -> str:
        """Apply tone-specific formatting."""
        
        if tone == ResponseTone.EMPATHETIC:
            # Add empathetic opening if not present
            empathetic_starters = ["I understand", "I'm sorry", "I can help", "I see"]
            if not any(starter in content for starter in empathetic_starters):
                content = f"I understand your concern. {content}"
        
        elif tone == ResponseTone.TECHNICAL:
            # Ensure technical precision
            if "issue" in content.lower() or "problem" in content.lower():
                content = content.replace("issue", "technical issue")
        
        elif tone == ResponseTone.URGENT:
            # Add urgency indicators
            content = f"**Important:** {content}"
        
        return content
    
    def _apply_channel_formatting(self, content: str, channel: str) -> tuple[str, bool]:
        """Apply channel-specific formatting optimizations."""
        
        channel_optimized = False
        
        if channel.lower() == "whatsapp":
            # WhatsApp optimizations
            # Remove markdown formatting that doesn't work well
            content = re.sub(r'\*\*(.*?)\*\*', r'*\1*', content)  # Bold to italic
            content = re.sub(r'\n\n+', '\n\n', content)  # Reduce excessive line breaks
            
            # Keep messages concise for mobile
            if len(content) > 500:
                sentences = content.split('. ')
                if len(sentences) > 3:
                    content = '. '.join(sentences[:3]) + '. Let me know if you need more details!'
            
            channel_optimized = True
            
        elif channel.lower() == "email":
            # Email optimizations
            content = f"Dear Customer,\n\n{content}\n\nBest regards,\nSAIA Support Team"
            channel_optimized = True
        
        return content, channel_optimized
    
    def _add_helpful_elements(
        self, 
        content: str, 
        category: str, 
        confidence: float, 
        sources_count: int
    ) -> str:
        """Add helpful elements to enhance user experience."""
        
        # Add category-specific helpful endings
        helpful_endings = {
            "troubleshooting": "If this doesn't resolve the issue, please let me know what happens when you try these steps.",
            "billing": "If you have any other billing questions, I'm here to help.",
            "setup": "Let me know if you need help with any of these steps!",
            "policies": "If you need clarification on any policy details, feel free to ask.",
            "general": "Is there anything else I can help you with?"
        }
        
        # Add helpful ending if not already present
        ending = helpful_endings.get(category, helpful_endings["general"])
        if not any(phrase in content.lower() for phrase in ["let me know", "feel free", "anything else"]):
            content = f"{content}\n\n{ending}"
        
        return content


# Global instance
_response_formatter = None


def get_response_formatter() -> ResponseFormatter:
    """Get global response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatter()
    return _response_formatter
