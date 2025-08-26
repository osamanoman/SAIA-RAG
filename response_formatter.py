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
        # All other channels get identical formatting for consistency
        return content, False
    
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
        """Format Arabic content for WhatsApp with comprehensive formatting and RTL support."""
        # Clean up the content first
        content = content.strip()
        
        # WhatsApp formatting best practices:
        # 1. Consistent bullet points
        # 2. Proper line spacing
        # 3. RTL text direction
        # 4. Strategic emoji usage
        # 5. Mobile-optimized structure
        
        # First, try to detect if content contains multiple services/points that should be bulleted
        bullet_indicators = [
            'خدمات', 'خدمة', 'مميزات', 'ميزات', 'خصائص', 'أقسام', 'أنواع', 'فئات',
            'أولاً', 'ثانياً', 'ثالثاً', 'أخيراً', 'أيضاً', 'بالإضافة إلى', 'كذلك',
            'مثل', 'تشمل', 'تتضمن', 'تقدم', 'توفر', 'تشمل', 'تتضمن', 'تقدم', 'توفر'
        ]
        
        has_bullet_indicators = any(indicator in content for indicator in bullet_indicators)
        
        # Clean up inconsistent bullet formatting first
        # Replace all variations of bullet points with consistent ones
        content = re.sub(r'[.*\-]\s*', '• ', content)  # Replace . * - with •
        content = re.sub(r'\.\s*\.\s*', '• ', content)  # Replace .. with •
        content = re.sub(r'\*\s*', '• ', content)  # Replace * with •
        content = re.sub(r'-\s*', '• ', content)  # Replace - with •
        
        # Clean up multiple consecutive bullet points
        content = re.sub(r'•\s*•\s*•\s*', '• ', content)  # Replace • • • with single •
        content = re.sub(r'•\s*•\s*', '• ', content)  # Replace • • with single •
        
        # Clean up bullet points followed by colons (convert to headers)
        content = re.sub(r'•\s*([^:]+):\s*•', r'**\1:**', content)  # Convert "• Service: •" to "**Service:**"
        
        # Split into lines
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains numbered or bullet points
            if re.match(r'^\d+\.', line):
                # Convert numbered points to bullet format
                line = re.sub(r'^\d+\.\s*', '• ', line)
                formatted_lines.append(line)
            elif re.match(r'^•', line):
                # Already formatted, keep as is
                formatted_lines.append(line)
            else:
                # Regular text line - check if it should be converted to bullet point
                if has_bullet_indicators and len(line) > 20:
                    # Look for sentence endings that suggest separate points
                    if any(ending in line for ending in ['،', '.', '؛', '!', '؟']):
                        # Split by common separators and create bullet points
                        sentences = re.split(r'[،.؛!؟]', line)
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence and len(sentence) > 10:  # Only meaningful sentences
                                # Ensure proper RTL formatting for Arabic
                                formatted_lines.append(f"• {sentence}")
                        continue
                
                # If not converting to bullets, keep as regular line
                formatted_lines.append(line)
        
        # Join lines with proper spacing for RTL and WhatsApp readability
        formatted_content = '\n\n'.join(formatted_lines)  # Double line spacing for better readability
        
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
        
        # Apply advanced WhatsApp formatting
        formatted_content = self._apply_whatsapp_advanced_formatting(formatted_content, "ar")
        
        # Ensure proper RTL formatting for Arabic text
        # Add RTL mark at the beginning to ensure proper text direction
        formatted_content = f"\u202B{formatted_content}"
        
        return formatted_content
    
    def _format_english_for_whatsapp(self, content: str) -> str:
        """Format English content for WhatsApp with comprehensive formatting."""
        # Clean up the content first
        content = content.strip()
        
        # WhatsApp formatting best practices:
        # 1. Consistent bullet points
        # 2. Proper line spacing
        # 3. Strategic emoji usage
        # 4. Mobile-optimized structure
        # 5. Clear hierarchy
        
        # First, try to detect if content contains multiple services/points that should be bulleted
        bullet_indicators = [
            'services', 'service', 'features', 'benefits', 'options', 'types', 'categories',
            'first', 'second', 'third', 'also', 'additionally', 'furthermore', 'moreover',
            'such as', 'including', 'provides', 'offers', 'enables'
        ]
        
        has_bullet_indicators = any(indicator in content.lower() for indicator in bullet_indicators)
        
        # Clean up inconsistent bullet formatting first
        # Replace all variations of bullet points with consistent ones
        content = re.sub(r'[.*\-]\s*', '• ', content)  # Replace . * - with •
        content = re.sub(r'\.\s*\.\s*', '• ', content)  # Replace .. with •
        content = re.sub(r'\*\s*', '• ', content)  # Replace * with •
        content = re.sub(r'-\s*', '• ', content)  # Replace - with •
        
        # Clean up multiple consecutive bullet points
        content = re.sub(r'•\s*•\s*•\s*', '• ', content)  # Replace • • • with single •
        content = re.sub(r'•\s*•\s*', '• ', content)  # Replace • • with single •
        
        # Clean up bullet points followed by colons (convert to headers)
        content = re.sub(r'•\s*([^:]+):\s*•', r'**\1:**', content)  # Convert "• Service: •" to "**Service:**"
        
        # Split into lines
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains numbered or bullet points
            if re.match(r'^\d+\.', line):
                # Convert numbered points to bullet format
                line = re.sub(r'^\d+\.\s*', '• ', line)
                formatted_lines.append(line)
            elif re.match(r'^•', line):
                # Already formatted, keep as is
                formatted_lines.append(line)
            else:
                # Regular text line - check if it should be converted to bullet point
                if has_bullet_indicators and len(line) > 20:
                    # Look for sentence endings that suggest separate points
                    if any(ending in line for ending in [',', '.', ';', '!', '?']):
                        # Split by common separators and create bullet points
                        sentences = re.split(r'[,.;!?]', line)
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence and len(sentence) > 10:  # Only meaningful sentences
                                formatted_lines.append(f"• {sentence}")
                        continue
                
                # If not converting to bullets, keep as regular line
                formatted_lines.append(line)
        
        # Join lines with proper spacing for WhatsApp readability
        formatted_content = '\n\n'.join(formatted_lines)  # Double line spacing for better readability
        
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
        
        # Apply advanced WhatsApp formatting
        formatted_content = self._apply_whatsapp_advanced_formatting(formatted_content, "en")
        
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

    def _apply_whatsapp_advanced_formatting(self, content: str, language: str) -> str:
        """
        Apply comprehensive WhatsApp formatting following best practices:
        
        1. Break text into shorter sentences
        2. Use bullet points or lists
        3. Use line breaks to separate ideas
        4. Add formatting (Bold, Italics)
        5. Improve readability using emojis
        6. Clarity in action points
        7. Context-specific responses
        """
        
        if language == "ar":
            # Arabic formatting improvements
            
            # 1. Add emojis for different sections
            content = re.sub(r'^(خدمات|مميزات|خصائص|أقسام|أنواع|فئات):', r'💼 \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(أولاً|ثانياً|ثالثاً|أخيراً):', r'🔢 \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(الميزات|الأسعار|طرق الدعم):', r'⭐ \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(تأمين|حماية|ضمان):', r'🛡️ \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(دعم|مساعدة|خدمة):', r'💬 \1:', content, flags=re.MULTILINE)
            
            # 2. Add emojis to bullet points for visual appeal
            content = re.sub(r'^\s*•\s*', '✅ ', content, flags=re.MULTILINE)
            
            # 3. Bold important headers
            content = re.sub(r'^(💼 [^:]+):', r'*\1*', content, flags=re.MULTILINE)
            content = re.sub(r'^(🔢 [^:]+):', r'*\1*', content, flags=re.MULTILINE)
            content = re.sub(r'^(⭐ [^:]+):', r'*\1*', content, flags=re.MULTILINE)
            
            # 4. Add line breaks for better readability
            content = re.sub(r'([.!؟])\s*', r'\1\n\n', content)
            
            # 5. Ensure proper spacing around bullet points
            content = re.sub(r'(\n✅ )', r'\n\n✅ ', content)
            
            # 6. Add emojis for common insurance terms
            content = re.sub(r'\b(تأمين المركبات)\b', r'🚗 \1', content)
            content = re.sub(r'\b(تأمين شامل)\b', r'🛡️ \1', content)
            content = re.sub(r'\b(وثائق التأمين)\b', r'📋 \1', content)
            content = re.sub(r'\b(المطالبات)\b', r'📝 \1', content)
            content = re.sub(r'\b(الدفع)\b', r'💳 \1', content)
            
        else:
            # English formatting improvements
            
            # 1. Add emojis for different sections
            content = re.sub(r'^(Services|Features|Benefits|Options|Types|Categories):', r'💼 \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(First|Second|Third|Finally):', r'🔢 \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(Features|Pricing|Support):', r'⭐ \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(Insurance|Protection|Coverage):', r'🛡️ \1:', content, flags=re.MULTILINE)
            content = re.sub(r'^(Support|Help|Service):', r'💬 \1:', content, flags=re.MULTILINE)
            
            # 2. Add emojis to bullet points
            content = re.sub(r'^\s*•\s*', '✅ ', content, flags=re.MULTILINE)
            
            # 3. Bold important headers
            content = re.sub(r'^(💼 [^:]+):', r'*\1*', content, flags=re.MULTILINE)
            content = re.sub(r'^(🔢 [^:]+):', r'*\1*', content, flags=re.MULTILINE)
            content = re.sub(r'^(⭐ [^:]+):', r'*\1*', content, flags=re.MULTILINE)
            
            # 4. Add line breaks for better readability
            content = re.sub(r'([.!?])\s*', r'\1\n\n', content)
            
            # 5. Ensure proper spacing around bullet points
            content = re.sub(r'(\n✅ )', r'\n\n✅ ', content)
            
            # 6. Add emojis for common insurance terms
            content = re.sub(r'\b(Vehicle Insurance)\b', r'🚗 \1', content, flags=re.IGNORECASE)
            content = re.sub(r'\b(Comprehensive Insurance)\b', r'🛡️ \1', content, flags=re.IGNORECASE)
            content = re.sub(r'\b(Insurance Documents)\b', r'📋 \1', content, flags=re.IGNORECASE)
            content = re.sub(r'\b(Claims)\b', r'📝 \1', content, flags=re.IGNORECASE)
            content = re.sub(r'\b(Payment)\b', r'💳 \1', content, flags=re.IGNORECASE)
        
        # 7. Limit line length for mobile screens
        max_line_length = 40 if language == "ar" else 60
        
        # Split long lines for better mobile readability
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if len(line) > max_line_length and not line.startswith('✅'):
                # Split long lines at natural break points
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) <= max_line_length:
                        current_line += (word + " ")
                    else:
                        if current_line:
                            formatted_lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    formatted_lines.append(current_line.strip())
            else:
                formatted_lines.append(line)
        
        # Join with proper spacing and ensure clean formatting
        formatted_content = '\n'.join(formatted_lines)
        
        # Final cleanup: ensure consistent spacing
        formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)  # Remove excessive line breaks
        formatted_content = re.sub(r'✅\s*✅', '✅', formatted_content)  # Remove duplicate checkmarks
        
        return formatted_content


# Global instance
_response_formatter = None


def get_response_formatter() -> ResponseFormatter:
    """Get global response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatter()
    return _response_formatter
