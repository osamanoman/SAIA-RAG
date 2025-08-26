import re
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class FormattedResponse:
    """Response object that provides the interface expected by rag_service.py"""
    
    def __init__(self, content: str, tone: str = "professional", format_type: str = "plain", channel_optimized: bool = True):
        self.content = content
        self.tone = type('Tone', (), {'value': tone})()
        self.format_type = type('FormatType', (), {'value': format_type})()
        self.channel_optimized = channel_optimized


class ResponseFormatter:
    """
    Simple response formatter that maintains the existing interface.
    Enhanced with WhatsApp-specific formatting improvements.
    """
    
    def __init__(self):
        self.logger = logger
    
    def format_response(
        self, 
        content: str, 
        channel: str = "unified", 
        tone: str = "professional",
        language: str = "auto",
        category: str = "general",
        confidence: float = 0.0,
        sources_count: int = 0,
        query_intent: str = "question"
    ) -> FormattedResponse:
        """
        Format response based on channel and tone.
        Returns a FormattedResponse object for compatibility.
        """
        if not content:
            return FormattedResponse("", tone, "plain", False)
            
        # Auto-detect language if not specified
        if language == "auto":
            language = self._detect_language(content)
        
        # Apply channel-specific formatting
        if channel == "whatsapp":
            formatted_content = self._format_for_whatsapp(content, language)
        else:
            formatted_content = content  # Keep original for non-WhatsApp channels
        
        # Return FormattedResponse object for compatibility
        return FormattedResponse(
            content=formatted_content,
            tone=tone,
            format_type="plain",
            channel_optimized=(channel == "whatsapp")
        )
    
    def _detect_language(self, content: str) -> str:
        """Detect if content is Arabic or English."""
        # Check for Arabic characters
        arabic_chars = re.findall(r'[\u0600-\u06FF]', content)
        if len(arabic_chars) > len(content) * 0.3:  # If more than 30% is Arabic
            return "ar"
        return "en"
    
    def _format_for_whatsapp(self, content: str, language: str) -> str:
        """Apply WhatsApp-specific formatting improvements."""
        if not content:
            return content
            
        # Clean up the content
        content = content.strip()
        
        if language == "ar":
            return self._format_arabic_for_whatsapp(content)
        else:
            return self._format_english_for_whatsapp(content)
    
    def _format_arabic_for_whatsapp(self, content: str) -> str:
        """Format Arabic content for WhatsApp with improved readability."""
        # Clean up multiple bullet points
        content = re.sub(r'•\s*•\s*•\s*', '• ', content)  # Replace • • • with single •
        content = re.sub(r'•\s*•\s*', '• ', content)  # Replace • • with single •
        
        # Ensure proper spacing around bullet points
        content = re.sub(r'•\s*', '\n\n✅ ', content)
        
        # Clean up excessive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Add RTL mark for Arabic
        if content.strip():
            content = f"\u202B{content}"
        
        return content
    
    def _format_english_for_whatsapp(self, content: str) -> str:
        """Format English content for WhatsApp with improved readability."""
        # Clean up multiple bullet points
        content = re.sub(r'•\s*•\s*•\s*', '• ', content)  # Replace • • • with single •
        content = re.sub(r'•\s*•\s*', '• ', content)  # Replace • • with single •
        
        # Ensure proper spacing around bullet points
        content = re.sub(r'•\s*', '\n\n✅ ', content)
        
        # Clean up excessive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content


# Global instance and getter function for compatibility
_response_formatter = ResponseFormatter()

def get_response_formatter() -> ResponseFormatter:
    """Get the global response formatter instance."""
    return _response_formatter
