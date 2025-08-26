import re
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class ResponseFormatter:
    """
    Formats RAG responses for different channels and tones.
    Optimized for WhatsApp with proper bullet points, spacing, and emojis.
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
    ) -> str:
        """
        Format response based on channel and tone.
        
        Args:
            content: Raw response content
            channel: Target channel (whatsapp, web, api)
            tone: Response tone (professional, friendly, casual)
            language: Content language (ar, en, auto)
        
        Returns:
            Formatted response string
        """
        if not content:
            return content
            
        # Auto-detect language if not specified
        if language == "auto":
            language = self._detect_language(content)
        
        # Apply channel-specific formatting
        if channel == "whatsapp":
            return self._format_for_whatsapp(content, language)
        elif channel == "web":
            return self._format_for_web(content, language)
        else:
            return self._format_unified(content, language)
    
    def _detect_language(self, content: str) -> str:
        """Detect if content is Arabic or English."""
        # Check for Arabic characters
        arabic_chars = re.findall(r'[\u0600-\u06FF]', content)
        if len(arabic_chars) > len(content) * 0.3:  # If more than 30% is Arabic
            return "ar"
        return "en"
    
    def _format_for_whatsapp(self, content: str, language: str) -> str:
        """Format content specifically for WhatsApp with enhanced formatting."""
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
            'مثل', 'تشمل', 'تتضمن', 'تقدم', 'توفر'
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
        
        # Fix the specific issue with "تشمل الخدمات:" format
        content = re.sub(r'\*\*([^:]+):\*\*', r'\n\n**💼 \1:**\n', content)
        
        # Ensure proper spacing around bullet points
        content = re.sub(r'•\s*', '\n\n✅ ', content)
        
        # Clean up excessive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
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
                line = re.sub(r'^\d+\.\s*', '✅ ', line)
                formatted_lines.append(line)
            elif re.match(r'^✅', line):
                # Already formatted bullet point
                formatted_lines.append(line)
            elif re.match(r'^\*\*', line):
                # Header line - add spacing
                formatted_lines.append(line)
            elif has_bullet_indicators and len(line) > 50:
                # Long line that might benefit from bullet points
                # Split by common Arabic separators
                sentences = re.split(r'[،.؛!؟]', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:  # Only meaningful sentences
                        # Ensure proper RTL formatting for Arabic
                        formatted_lines.append(f"✅ {sentence}")
                continue
            else:
                formatted_lines.append(line)
        
        # Join lines with proper spacing for RTL and WhatsApp readability
        formatted_content = '\n\n'.join(formatted_lines)  # Double line spacing for better readability
        
        # Add WhatsApp-friendly emojis for common patterns
        if 'تأمين' in content:
            formatted_content = "🛡️ " + formatted_content
        
        # Add list indicator if we have bullet points
        if any(line.startswith('✅') for line in formatted_lines):
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
            'first', 'second', 'third', 'finally', 'also', 'additionally', 'moreover',
            'such as', 'including', 'providing', 'offering', 'available'
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
        
        # Fix the specific issue with service lists format
        content = re.sub(r'\*\*([^:]+):\*\*', r'\n\n**💼 \1:**\n', content)
        
        # Ensure proper spacing around bullet points
        content = re.sub(r'•\s*', '\n\n✅ ', content)
        
        # Clean up excessive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
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
                line = re.sub(r'^\d+\.\s*', '✅ ', line)
                formatted_lines.append(line)
            elif re.match(r'^✅', line):
                # Already formatted bullet point
                formatted_lines.append(line)
            elif re.match(r'^\*\*', line):
                # Header line - add spacing
                formatted_lines.append(line)
            elif has_bullet_indicators and len(line) > 60:
                # Long line that might benefit from bullet points
                # Split by common English separators
                sentences = re.split(r'[,.;!?]', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:  # Only meaningful sentences
                        formatted_lines.append(f"✅ {sentence}")
                continue
            else:
                formatted_lines.append(line)
        
        # Join lines with proper spacing for WhatsApp readability
        formatted_content = '\n\n'.join(formatted_lines)  # Double line spacing for better readability
        
        # Add WhatsApp-friendly emojis for common patterns
        if 'insurance' in content.lower():
            formatted_content = "🛡️ " + formatted_content
        
        # Add list indicator if we have bullet points
        if any(line.startswith('✅') for line in formatted_lines):
            formatted_content = "📋 " + formatted_content
        
        # Apply advanced WhatsApp formatting
        formatted_content = self._apply_whatsapp_advanced_formatting(formatted_content, "en")
        
        return formatted_content
    
    def _format_for_web(self, content: str, language: str) -> str:
        """Format content for web display."""
        # Web formatting can be more complex, keep simple for now
        return content
    
    def _format_unified(self, content: str, language: str) -> str:
        """Format content for general use."""
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
            content = re.sub(r'^\s*✅\s*', '✅ ', content, flags=re.MULTILINE)
            
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
            content = re.sub(r'^\s*✅\s*', '✅ ', content, flags=re.MULTILINE)
            
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


class FormattedResponse:
    """Response object that mimics the old formatter's response structure."""
    
    def __init__(self, content: str, tone: str = "professional", format_type: str = "plain", channel_optimized: bool = True):
        self.content = content
        self.tone = type('Tone', (), {'value': tone})()
        self.format_type = type('FormatType', (), {'value': format_type})()
        self.channel_optimized = channel_optimized


# Global instance and getter function for compatibility
_response_formatter = ResponseFormatter()

def get_response_formatter() -> ResponseFormatter:
    """Get the global response formatter instance."""
    return _response_formatter
