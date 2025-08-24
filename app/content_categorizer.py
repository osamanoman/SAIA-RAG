"""
SAIA-RAG Content Categorization Module

Intelligent content categorization system for customer support documents.
Automatically categorizes content and enhances metadata for better retrieval routing.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .openai_client import get_openai_client

logger = structlog.get_logger()


class ContentCategory(str, Enum):
    """Standard customer support content categories."""
    FAQ = "FAQ"
    TROUBLESHOOTING = "troubleshooting"
    BILLING = "billing"
    SETUP = "setup"
    POLICIES = "policies"
    TERMS_CONDITIONS = "terms and conditions"
    SERVICES = "services"
    SUPPORT = "support"


class ContentIntent(str, Enum):
    """Content intent types for better routing."""
    QUESTION = "question"
    PROCEDURE = "procedure"
    POLICY = "policy"
    DEFINITION = "definition"
    COMPLAINT = "complaint"
    REQUEST = "request"
    INFORMATION = "information"


class CategoryConfidence(BaseModel):
    """Category prediction with confidence score."""
    category: ContentCategory = Field(..., description="Predicted category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning for categorization")


class EnhancedMetadata(BaseModel):
    """Enhanced metadata with intelligent categorization."""
    primary_category: ContentCategory = Field(..., description="Primary content category")
    secondary_categories: List[ContentCategory] = Field(default_factory=list, description="Secondary categories")
    content_intent: ContentIntent = Field(..., description="Content intent type")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    auto_generated_tags: List[str] = Field(default_factory=list, description="Auto-generated tags")
    language: str = Field(..., description="Content language (ar/en)")
    complexity_level: str = Field(..., description="Content complexity (basic/intermediate/advanced)")
    target_audience: str = Field(..., description="Target audience (customer/admin/technical)")
    urgency_level: str = Field(..., description="Urgency level (low/medium/high)")
    categorization_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentCategorizer:
    """
    Intelligent content categorization system for customer support.
    
    Features:
    - Multi-language support (Arabic/English)
    - Intent detection and classification
    - Confidence scoring and validation
    - Automatic tag generation
    - Category mapping and normalization
    """
    
    def __init__(self):
        """Initialize content categorizer."""
        self.settings = get_settings()
        self.openai_client = get_openai_client()
        
        # Category mapping for existing content
        self.category_mappings = {
            # Arabic categories to English
            "سياسة الخصوصية": ContentCategory.POLICIES,
            "شروط و احكام": ContentCategory.TERMS_CONDITIONS,
            "خدمات": ContentCategory.SERVICES,
            "عن الشركة": ContentCategory.SUPPORT,
            "insurance": ContentCategory.SERVICES,
            "FAQ": ContentCategory.FAQ,
            
            # English variations
            "privacy policy": ContentCategory.POLICIES,
            "terms and conditions": ContentCategory.TERMS_CONDITIONS,
            "terms & conditions": ContentCategory.TERMS_CONDITIONS,
            "services": ContentCategory.SERVICES,
            "about": ContentCategory.SUPPORT,
            "support": ContentCategory.SUPPORT,
            "help": ContentCategory.SUPPORT,
            "billing": ContentCategory.BILLING,
            "payment": ContentCategory.BILLING,
            "troubleshooting": ContentCategory.TROUBLESHOOTING,
            "setup": ContentCategory.SETUP,
            "installation": ContentCategory.SETUP
        }
        
        # Content patterns for classification
        self.classification_patterns = {
            ContentCategory.FAQ: {
                "keywords": [
                    "سؤال", "جواب", "كيف", "ما هو", "هل يمكن", "لماذا",
                    "question", "answer", "how", "what", "can", "why", "faq"
                ],
                "patterns": [
                    r"سؤال\s*\d*\s*[:：]",
                    r"س\s*\d*\s*[:：]",
                    r"جواب\s*\d*\s*[:：]",
                    r"ج\s*\d*\s*[:：]",
                    r"Q\s*\d*\s*[:：]",
                    r"A\s*\d*\s*[:：]"
                ],
                "confidence_boost": 0.3
            },
            ContentCategory.TROUBLESHOOTING: {
                "keywords": [
                    "مشكلة", "خطأ", "لا يعمل", "عطل", "إصلاح", "حل",
                    "problem", "error", "issue", "fix", "solve", "not working", "broken"
                ],
                "patterns": [
                    r"خطأ\s+رقم",
                    r"error\s+code",
                    r"لا\s+يعمل",
                    r"not\s+working",
                    r"مشكلة\s+في"
                ],
                "confidence_boost": 0.4
            },
            ContentCategory.BILLING: {
                "keywords": [
                    "دفع", "فاتورة", "سعر", "تكلفة", "رسوم", "مبلغ", "استرداد",
                    "payment", "billing", "invoice", "cost", "price", "fee", "refund", "charge"
                ],
                "patterns": [
                    r"كيف\s+أدفع",
                    r"how\s+to\s+pay",
                    r"رسوم\s+الخدمة",
                    r"service\s+fee",
                    r"استرداد\s+المبلغ"
                ],
                "confidence_boost": 0.4
            },
            ContentCategory.SETUP: {
                "keywords": [
                    "تسجيل", "إعداد", "تثبيت", "تكوين", "بدء", "تشغيل",
                    "setup", "install", "configure", "register", "start", "begin", "create account"
                ],
                "patterns": [
                    r"خطوات\s+التسجيل",
                    r"registration\s+steps",
                    r"كيفية\s+الإعداد",
                    r"how\s+to\s+setup"
                ],
                "confidence_boost": 0.3
            },
            ContentCategory.POLICIES: {
                "keywords": [
                    "سياسة", "خصوصية", "بيانات", "معلومات شخصية", "حماية",
                    "policy", "privacy", "data", "personal information", "protection"
                ],
                "patterns": [
                    r"سياسة\s+الخصوصية",
                    r"privacy\s+policy",
                    r"حماية\s+البيانات",
                    r"data\s+protection"
                ],
                "confidence_boost": 0.4
            },
            ContentCategory.TERMS_CONDITIONS: {
                "keywords": [
                    "شروط", "أحكام", "اتفاقية", "التزامات", "قوانين",
                    "terms", "conditions", "agreement", "obligations", "rules"
                ],
                "patterns": [
                    r"شروط\s+الاستخدام",
                    r"terms\s+of\s+use",
                    r"الشروط\s+والأحكام",
                    r"terms\s+and\s+conditions"
                ],
                "confidence_boost": 0.4
            },
            ContentCategory.SERVICES: {
                "keywords": [
                    "خدمة", "خدمات", "منتج", "منتجات", "عرض", "باقة",
                    "service", "services", "product", "products", "offer", "package"
                ],
                "patterns": [
                    r"خدمات\s+الشركة",
                    r"company\s+services",
                    r"أنواع\s+الخدمات",
                    r"types\s+of\s+services"
                ],
                "confidence_boost": 0.3
            }
        }
        
        # Intent detection patterns
        self.intent_patterns = {
            ContentIntent.QUESTION: [
                r"[؟?]", r"كيف", r"ما هو", r"هل", r"لماذا", r"متى", r"أين",
                r"how", r"what", r"can", r"why", r"when", r"where"
            ],
            ContentIntent.PROCEDURE: [
                r"خطوة", r"أولاً", r"ثانياً", r"ثالثاً", r"اتبع", r"قم بـ",
                r"step", r"first", r"then", r"next", r"follow", r"do"
            ],
            ContentIntent.POLICY: [
                r"سياسة", r"قانون", r"قاعدة", r"نظام",
                r"policy", r"rule", r"regulation", r"law"
            ],
            ContentIntent.DEFINITION: [
                r"تعريف", r"معنى", r"هو", r"يعني",
                r"definition", r"meaning", r"means", r"is"
            ]
        }
        
        logger.info("Content categorizer initialized")
    
    async def categorize_content(
        self,
        text: str,
        title: Optional[str] = None,
        existing_category: Optional[str] = None,
        existing_tags: Optional[List[str]] = None
    ) -> EnhancedMetadata:
        """
        Categorize content with enhanced metadata.
        
        Args:
            text: Content text to categorize
            title: Document title
            existing_category: Current category (if any)
            existing_tags: Current tags (if any)
            
        Returns:
            Enhanced metadata with categorization
        """
        try:
            # Detect language
            language = self._detect_language(text)
            
            # Get category predictions
            category_predictions = self._predict_categories(text, title)
            
            # Select primary category
            primary_category = self._select_primary_category(
                category_predictions, existing_category
            )
            
            # Detect content intent
            content_intent = self._detect_intent(text)
            
            # Generate enhanced tags
            auto_tags = self._generate_tags(text, title, primary_category)
            
            # Determine complexity and audience
            complexity = self._assess_complexity(text)
            audience = self._determine_audience(text, primary_category)
            urgency = self._assess_urgency(text, primary_category)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(category_predictions)
            
            # Get secondary categories
            secondary_categories = [
                pred.category for pred in category_predictions[1:3]
                if pred.confidence > 0.3 and pred.category != primary_category
            ]
            
            enhanced_metadata = EnhancedMetadata(
                primary_category=primary_category,
                secondary_categories=secondary_categories,
                content_intent=content_intent,
                confidence_score=confidence,
                auto_generated_tags=auto_tags,
                language=language,
                complexity_level=complexity,
                target_audience=audience,
                urgency_level=urgency,
                categorization_metadata={
                    "categorized_at": datetime.utcnow().isoformat(),
                    "method": "intelligent_classification",
                    "predictions": [
                        {
                            "category": pred.category.value,
                            "confidence": pred.confidence,
                            "reasoning": pred.reasoning
                        }
                        for pred in category_predictions[:3]
                    ]
                }
            )
            
            logger.info(
                "Content categorized successfully",
                primary_category=primary_category.value,
                confidence=confidence,
                language=language,
                intent=content_intent.value
            )
            
            return enhanced_metadata
            
        except Exception as e:
            logger.error("Content categorization failed", error=str(e))
            # Return fallback categorization
            return self._fallback_categorization(text, existing_category)
    
    def _predict_categories(
        self,
        text: str,
        title: Optional[str] = None
    ) -> List[CategoryConfidence]:
        """Predict categories using pattern matching and ML."""
        predictions = []
        
        # Combine text and title for analysis
        full_text = f"{title or ''} {text}".lower()
        
        for category, config in self.classification_patterns.items():
            confidence = 0.0
            reasoning_parts = []
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in full_text)
            if keyword_matches > 0:
                keyword_confidence = min(0.4, keyword_matches * 0.1)
                confidence += keyword_confidence
                reasoning_parts.append(f"{keyword_matches} keyword matches")
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in config["patterns"] if re.search(pattern, full_text, re.IGNORECASE))
            if pattern_matches > 0:
                pattern_confidence = min(0.3, pattern_matches * 0.15)
                confidence += pattern_confidence
                reasoning_parts.append(f"{pattern_matches} pattern matches")
            
            # Apply confidence boost
            if confidence > 0:
                confidence = min(1.0, confidence + config.get("confidence_boost", 0))
            
            # Create reasoning
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No strong indicators"
            
            predictions.append(CategoryConfidence(
                category=category,
                confidence=confidence,
                reasoning=reasoning
            ))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions

    def _select_primary_category(
        self,
        predictions: List[CategoryConfidence],
        existing_category: Optional[str] = None
    ) -> ContentCategory:
        """Select the primary category from predictions."""
        if not predictions:
            return ContentCategory.SUPPORT  # Default fallback

        # If existing category is valid and has reasonable confidence, prefer it
        if existing_category:
            mapped_category = self.category_mappings.get(existing_category.lower())
            if mapped_category:
                # Find prediction for existing category
                for pred in predictions:
                    if pred.category == mapped_category and pred.confidence > 0.2:
                        return mapped_category

        # Otherwise, use highest confidence prediction
        return predictions[0].category

    def _detect_intent(self, text: str) -> ContentIntent:
        """Detect content intent from text patterns."""
        text_lower = text.lower()

        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            return ContentIntent.INFORMATION  # Default

        # Return intent with highest score
        return max(intent_scores.items(), key=lambda x: x[1])[0]

    def _generate_tags(
        self,
        text: str,
        title: Optional[str],
        category: ContentCategory
    ) -> List[str]:
        """Generate relevant tags for content."""
        tags = set()

        # Add category-based tags
        category_tags = {
            ContentCategory.FAQ: ["faq", "questions", "answers"],
            ContentCategory.TROUBLESHOOTING: ["troubleshooting", "problems", "solutions"],
            ContentCategory.BILLING: ["billing", "payments", "costs"],
            ContentCategory.SETUP: ["setup", "installation", "configuration"],
            ContentCategory.POLICIES: ["policies", "privacy", "data"],
            ContentCategory.TERMS_CONDITIONS: ["terms", "conditions", "legal"],
            ContentCategory.SERVICES: ["services", "products", "offerings"],
            ContentCategory.SUPPORT: ["support", "help", "assistance"]
        }

        tags.update(category_tags.get(category, []))

        # Extract keywords from text
        text_lower = (title or "").lower() + " " + text.lower()

        # Common customer support keywords
        keyword_patterns = {
            "account": ["حساب", "account", "profile"],
            "password": ["كلمة مرور", "password", "login"],
            "payment": ["دفع", "payment", "pay"],
            "insurance": ["تأمين", "insurance"],
            "vehicle": ["مركبة", "vehicle", "car"],
            "policy": ["وثيقة", "policy"],
            "claim": ["مطالبة", "claim"],
            "renewal": ["تجديد", "renewal"],
            "registration": ["تسجيل", "registration"],
            "documents": ["وثائق", "documents", "papers"]
        }

        for tag, keywords in keyword_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.add(tag)

        # Language-specific tags
        if self._detect_language(text) == "ar":
            tags.add("arabic")
        else:
            tags.add("english")

        return list(tags)[:10]  # Limit to 10 tags

    def _detect_language(self, text: str) -> str:
        """Detect text language (Arabic or English)."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))

        if total_chars == 0:
            return "en"  # Default to English

        arabic_ratio = arabic_chars / total_chars
        return "ar" if arabic_ratio > 0.5 else "en"

    def _assess_complexity(self, text: str) -> str:
        """Assess content complexity level."""
        # Simple heuristics for complexity
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?؟]', text))
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Technical terms indicators
        technical_terms = [
            "api", "configuration", "database", "server", "protocol",
            "تكوين", "قاعدة بيانات", "خادم", "بروتوكول"
        ]

        technical_count = sum(1 for term in technical_terms if term in text.lower())

        if technical_count > 2 or avg_sentence_length > 20:
            return "advanced"
        elif word_count > 200 or avg_sentence_length > 15:
            return "intermediate"
        else:
            return "basic"

    def _determine_audience(self, text: str, category: ContentCategory) -> str:
        """Determine target audience for content."""
        text_lower = text.lower()

        # Admin/technical indicators
        admin_indicators = [
            "configuration", "admin", "system", "database", "server",
            "إعداد النظام", "إدارة", "نظام", "قاعدة بيانات"
        ]

        # Technical indicators
        technical_indicators = [
            "api", "code", "programming", "development", "technical",
            "برمجة", "تطوير", "تقني", "كود"
        ]

        if any(indicator in text_lower for indicator in admin_indicators):
            return "admin"
        elif any(indicator in text_lower for indicator in technical_indicators):
            return "technical"
        elif category in [ContentCategory.POLICIES, ContentCategory.TERMS_CONDITIONS]:
            return "general"
        else:
            return "customer"

    def _assess_urgency(self, text: str, category: ContentCategory) -> str:
        """Assess content urgency level."""
        text_lower = text.lower()

        # High urgency indicators
        high_urgency = [
            "urgent", "emergency", "critical", "immediate", "asap",
            "عاجل", "طارئ", "فوري", "حرج"
        ]

        # Medium urgency indicators
        medium_urgency = [
            "important", "priority", "soon", "quickly",
            "مهم", "أولوية", "سريع", "عاجل"
        ]

        if any(indicator in text_lower for indicator in high_urgency):
            return "high"
        elif any(indicator in text_lower for indicator in medium_urgency):
            return "medium"
        elif category == ContentCategory.TROUBLESHOOTING:
            return "medium"  # Troubleshooting is generally medium priority
        else:
            return "low"

    def _calculate_overall_confidence(self, predictions: List[CategoryConfidence]) -> float:
        """Calculate overall confidence score."""
        if not predictions:
            return 0.0

        # Use highest confidence with some adjustment for second-best
        primary_confidence = predictions[0].confidence

        if len(predictions) > 1:
            secondary_confidence = predictions[1].confidence
            # If second prediction is very close, reduce overall confidence
            if secondary_confidence > primary_confidence * 0.8:
                return primary_confidence * 0.9

        return primary_confidence

    def _fallback_categorization(
        self,
        text: str,
        existing_category: Optional[str] = None
    ) -> EnhancedMetadata:
        """Provide fallback categorization when main process fails."""
        # Try to map existing category
        primary_category = ContentCategory.SUPPORT
        if existing_category:
            mapped = self.category_mappings.get(existing_category.lower())
            if mapped:
                primary_category = mapped

        return EnhancedMetadata(
            primary_category=primary_category,
            secondary_categories=[],
            content_intent=ContentIntent.INFORMATION,
            confidence_score=0.5,
            auto_generated_tags=["fallback"],
            language=self._detect_language(text),
            complexity_level="basic",
            target_audience="customer",
            urgency_level="low",
            categorization_metadata={
                "categorized_at": datetime.utcnow().isoformat(),
                "method": "fallback",
                "reason": "Main categorization failed"
            }
        )

    async def batch_categorize_knowledge_base(self) -> Dict[str, Any]:
        """
        Batch categorize all content in the knowledge base.

        Returns:
            Summary of categorization results
        """
        try:
            from .vector_store import get_vector_store

            logger.info("Starting batch categorization of knowledge base")
            start_time = datetime.utcnow()

            vs = get_vector_store()

            # Get all points from vector store
            all_points = []
            offset = None

            while True:
                points, next_offset = vs.client.scroll(
                    collection_name=vs.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                all_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset

            # Categorize each point
            categorization_results = []
            category_counts = {}
            confidence_scores = []

            for point in all_points:
                payload = point.payload
                text = payload.get('text', '')
                title = payload.get('title')
                existing_category = payload.get('category')

                # Categorize content
                enhanced_metadata = await self.categorize_content(
                    text=text,
                    title=title,
                    existing_category=existing_category
                )

                # Track results
                category = enhanced_metadata.primary_category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                confidence_scores.append(enhanced_metadata.confidence_score)

                categorization_results.append({
                    "point_id": point.id,
                    "original_category": existing_category,
                    "new_category": category,
                    "confidence": enhanced_metadata.confidence_score,
                    "intent": enhanced_metadata.content_intent.value,
                    "language": enhanced_metadata.language,
                    "tags": enhanced_metadata.auto_generated_tags
                })

            # Calculate summary statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            summary = {
                "status": "completed",
                "processing_time_seconds": processing_time,
                "total_chunks_processed": len(all_points),
                "average_confidence": avg_confidence,
                "category_distribution": category_counts,
                "high_confidence_chunks": sum(1 for score in confidence_scores if score > 0.7),
                "low_confidence_chunks": sum(1 for score in confidence_scores if score < 0.5),
                "completed_at": end_time.isoformat()
            }

            logger.info(
                "Batch categorization completed",
                **summary
            )

            return {
                "summary": summary,
                "detailed_results": categorization_results
            }

        except Exception as e:
            logger.error("Batch categorization failed", error=str(e))
            raise

    def map_legacy_categories(self) -> Dict[str, str]:
        """
        Map legacy categories to new standard categories.

        Returns:
            Mapping dictionary
        """
        return {
            old_cat: new_cat.value
            for old_cat, new_cat in self.category_mappings.items()
        }

    def validate_category_coverage(self, content_samples: List[str]) -> Dict[str, Any]:
        """
        Validate category coverage for content samples.

        Args:
            content_samples: List of content samples to test

        Returns:
            Coverage analysis results
        """
        category_coverage = {cat.value: 0 for cat in ContentCategory}
        total_samples = len(content_samples)

        for sample in content_samples:
            predictions = self._predict_categories(sample)
            if predictions:
                primary_category = predictions[0].category.value
                category_coverage[primary_category] += 1

        # Calculate coverage percentages
        coverage_percentages = {
            cat: (count / total_samples) * 100
            for cat, count in category_coverage.items()
        }

        # Identify gaps
        low_coverage_categories = [
            cat for cat, percentage in coverage_percentages.items()
            if percentage < 5.0  # Less than 5% coverage
        ]

        return {
            "total_samples": total_samples,
            "category_coverage": category_coverage,
            "coverage_percentages": coverage_percentages,
            "low_coverage_categories": low_coverage_categories,
            "well_covered_categories": [
                cat for cat, percentage in coverage_percentages.items()
                if percentage > 20.0
            ]
        }

    def generate_categorization_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable categorization report."""
        summary = results.get("summary", {})

        report = f"""
# Content Categorization Report

**Processing Date**: {summary.get('completed_at', 'Unknown')}
**Total Chunks Processed**: {summary.get('total_chunks_processed', 0)}
**Processing Time**: {summary.get('processing_time_seconds', 0):.1f} seconds
**Average Confidence**: {summary.get('average_confidence', 0):.2f}

## Category Distribution
"""

        category_dist = summary.get('category_distribution', {})
        for category, count in sorted(category_dist.items()):
            percentage = (count / summary.get('total_chunks_processed', 1)) * 100
            report += f"- **{category}**: {count} chunks ({percentage:.1f}%)\n"

        report += f"""
## Quality Metrics
- **High Confidence Chunks** (>0.7): {summary.get('high_confidence_chunks', 0)}
- **Low Confidence Chunks** (<0.5): {summary.get('low_confidence_chunks', 0)}
- **Confidence Distribution**: {summary.get('average_confidence', 0):.1f} average

## Recommendations
"""

        low_confidence = summary.get('low_confidence_chunks', 0)
        total_chunks = summary.get('total_chunks_processed', 1)

        if low_confidence / total_chunks > 0.2:
            report += "- Consider reviewing low-confidence categorizations manually\n"

        if 'troubleshooting' not in category_dist or category_dist.get('troubleshooting', 0) == 0:
            report += "- Add more troubleshooting content to improve coverage\n"

        if 'billing' not in category_dist or category_dist.get('billing', 0) == 0:
            report += "- Add more billing-related content\n"

        return report


# Global instance
_content_categorizer = None


def get_content_categorizer() -> ContentCategorizer:
    """Get global content categorizer instance."""
    global _content_categorizer
    if _content_categorizer is None:
        _content_categorizer = ContentCategorizer()
    return _content_categorizer
