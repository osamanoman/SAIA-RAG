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
        
        # DISABLED: Empathetic tone formatting removed to prevent unwanted English text
        # if tone == ResponseTone.EMPATHETIC:
        #     # Only add empathetic opening for specific cases, not every response
        #     empathetic_starters = ["I understand", "I'm sorry", "I can help", "I see"]
        #     if not any(starter in content for starter in empathetic_starters):
        #         # Only add empathetic prefix for troubleshooting or complaint queries
        #         if query_intent in ["complaint", "troubleshooting", "urgent"]:
        #             content = f"I understand your concern. {content}"
        
        if tone == ResponseTone.TECHNICAL:
            # Ensure technical precision
            if "issue" in content.lower() or "problem" in content.lower():
                content = content.replace("issue", "technical issue")
        
        elif tone == ResponseTone.URGENT:
            # Add urgency indicators
            content = f"**Important:** {content}"
        
        return content
    
    def _apply_channel_formatting(self, content: str, channel: str) -> tuple[str, bool]:
        """Apply channel-specific formatting optimizations."""
        if channel == "whatsapp":
            return self._format_for_whatsapp(content), True
        elif channel == "web":
            return self._format_for_web(content), True
        # All other channels get identical formatting for consistency
        return content, False

    def _format_for_web(self, content: str) -> str:
        """
        Format content for web display with proper markdown structure.

        Converts inline markdown to multi-line markdown that marked.js can parse correctly.
        This fixes the issue where AI generates everything in one line without line breaks.
        """
        import re

        # Step 1: Add line breaks before bold headers that start sections
        # Pattern: "text **Header:**" -> "text\n\n**Header:**"
        # Do this first before handling numbered lists
        content = re.sub(r'([^\n])\s+(\*\*[^*]+:\*\*)', r'\1\n\n\2', content)

        # Step 2: Add line breaks before numbered list items (1. 2. 3. etc.)
        # Pattern: "text 1. **Title:**" -> "text\n\n1. **Title:**"
        # Keep the number with its content on the same line
        content = re.sub(r'([^\n\d])\s+(\d+\.\s+)', r'\1\n\n\2', content)

        # Step 2b: Merge numbered items that are separated from their content
        # Pattern: "1.\n\n**Title:**" -> "1. **Title:**"
        content = re.sub(r'(\d+\.)\s*\n+\s*(\*\*)', r'\1 \2', content)

        # Step 3: Add line breaks before bullet points (- or •)
        # Pattern: "text - item" or "text • item" -> "text\n\n- item"
        content = re.sub(r'([^\n])\s+([-•]\s+)', r'\1\n\n\2', content)

        # Step 4: Add line break after emoji at the start if followed by text
        # Pattern: "📋 text" -> "📋\n\ntext"
        content = re.sub(r'^([\U0001F300-\U0001F9FF])\s+', r'\1\n\n', content)

        # Step 5: Clean up multiple consecutive line breaks (more than 2)
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()
    
    def _format_for_whatsapp(self, content: str) -> str:
        """
        Format content specifically for WhatsApp display.
        
        WhatsApp formatting features:
        - Clean bullet points with • symbols
        - Proper spacing between sections
        - Emoji for visual appeal
        - Simple, readable structure
        """
        # Detect language
        language = self._detect_language(content)
        
        if language == "ar":
            return self._format_arabic_for_whatsapp(content)
        else:
            return self._format_english_for_whatsapp(content)
    
    def _format_arabic_for_whatsapp(self, content: str) -> str:
        """Format Arabic content for WhatsApp."""
        # Clean up the content first
        content = content.strip()
        
        # Split into lines
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains numbered or bullet points
            if re.match(r'^\d+\.', line) or re.match(r'^\*', line):
                # Convert to WhatsApp bullet format
                line = re.sub(r'^\d+\.\s*', '• ', line)
                line = re.sub(r'^\*\s*', '• ', line)
                formatted_lines.append(line)
            elif re.match(r'^•', line):
                # Already formatted, keep as is
                formatted_lines.append(line)
            else:
                # Regular text line
                formatted_lines.append(line)
        
        # Join lines with proper spacing
        formatted_content = '\n'.join(formatted_lines)
        
        # Add WhatsApp-friendly emojis for common patterns
        if 'خدمات' in formatted_content or 'خدمة' in formatted_content:
            formatted_content = "🛠️ " + formatted_content
        
        if 'مشكلة' in formatted_content or 'حل' in formatted_content:
            formatted_content = "🔧 " + formatted_content
            
        if 'إعداد' in formatted_content or 'تثبيت' in formatted_content:
            formatted_content = "⚙️ " + formatted_content
            
        if 'فواتير' in formatted_content or 'دفع' in formatted_content:
            formatted_content = "💰 " + formatted_content
            
        if 'سياسة' in formatted_content or 'شروط' in formatted_content:
            formatted_content = "📋 " + formatted_content
        
        return formatted_content
    
    def _format_english_for_whatsapp(self, content: str) -> str:
        """Format English content for WhatsApp."""
        # Clean up the content first
        content = content.strip()
        
        # Split into lines
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains numbered or bullet points
            if re.match(r'^\d+\.', line) or re.match(r'^\*', line):
                # Convert to WhatsApp bullet format
                line = re.sub(r'^\d+\.\s*', '• ', line)
                line = re.sub(r'^\*\s*', '• ', line)
                formatted_lines.append(line)
            elif re.match(r'^•', line):
                # Already formatted, keep as is
                formatted_lines.append(line)
            else:
                # Regular text line
                formatted_lines.append(line)
        
        # Join lines with proper spacing
        formatted_content = '\n'.join(formatted_lines)
        
        # Add WhatsApp-friendly emojis for common patterns
        if 'service' in formatted_content.lower() or 'help' in formatted_content.lower():
            formatted_content = "🛠️ " + formatted_content
        
        if 'problem' in formatted_content.lower() or 'issue' in formatted_content.lower():
            formatted_content = "🔧 " + formatted_content
            
        if 'setup' in formatted_content.lower() or 'install' in formatted_content.lower():
            formatted_content = "⚙️ " + formatted_content
            
        if 'billing' in formatted_content.lower() or 'payment' in formatted_content.lower():
            formatted_content = "💰 " + formatted_content
            
        if 'policy' in formatted_content.lower() or 'terms' in formatted_content.lower():
            formatted_content = "📋 " + formatted_content
        
        return formatted_content
    
    def _add_helpful_elements(
        self, 
        content: str, 
        category: str, 
        confidence: float, 
        sources_count: int
    ) -> str:
        """Add helpful elements to enhance user experience."""
        
        # Add category-specific helpful endings
        # Detect language from content
        language = self._detect_language(content)

        if language == "ar":
            helpful_endings = {
                "troubleshooting": "إذا لم يحل هذا المشكلة، يرجى إعلامي بما يحدث عند تجربة هذه الخطوات.",
                "billing": "إذا كان لديك أي أسئلة أخرى حول الفواتير، أنا هنا للمساعدة.",
                "setup": "أعلمني إذا كنت بحاجة للمساعدة في أي من هذه الخطوات!",
                "policies": "إذا كنت بحاجة لتوضيح أي تفاصيل في السياسة، لا تتردد في السؤال.",
                "general": "هل هناك أي شيء آخر يمكنني مساعدتك به؟"
            }
        else:
            helpful_endings = {
                "troubleshooting": "If this doesn't resolve the issue, please let me know what happens when you try these steps.",
                "billing": "If you have any other billing questions, I'm here to help.",
                "setup": "Let me know if you need help with any of these steps!",
                "policies": "If you need clarification on any policy details, feel free to ask.",
                "general": "Is there anything else I can help you with?"
            }
        
        # Disable helpful endings to maintain language consistency
        # The AI should generate complete responses without additional prompts
        # ending = helpful_endings.get(category, helpful_endings["general"])
        # if not any(phrase in content.lower() for phrase in ["let me know", "feel free", "anything else"]):
        #     content = f"{content}\n\n{ending}"
        
        return content


# Global instance
_response_formatter = None


def get_response_formatter() -> ResponseFormatter:
    """Get global response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatter()
    return _response_formatter
