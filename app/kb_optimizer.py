"""
SAIA-RAG Knowledge Base Optimizer

Handles knowledge base analysis, categorization, and optimization for customer support.
Implements best practices for content organization and metadata enhancement.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import Counter
import re

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .vector_store import get_vector_store

logger = structlog.get_logger()


class CategoryMapping(BaseModel):
    """Category mapping configuration."""
    old_category: str = Field(..., description="Original category name")
    new_category: str = Field(..., description="Standardized category name")
    confidence: float = Field(..., description="Mapping confidence (0.0-1.0)")
    reason: str = Field(..., description="Reason for mapping")


class ContentGap(BaseModel):
    """Content gap analysis result."""
    category: str = Field(..., description="Missing category")
    current_count: int = Field(..., description="Current chunk count")
    target_count: int = Field(..., description="Target chunk count")
    gap: int = Field(..., description="Gap to fill")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    suggested_topics: List[str] = Field(default_factory=list, description="Suggested content topics")


class KBAnalysisResult(BaseModel):
    """Knowledge base analysis result."""
    total_chunks: int = Field(..., description="Total chunks analyzed")
    total_documents: int = Field(..., description="Total unique documents")
    category_distribution: Dict[str, int] = Field(..., description="Current category distribution")
    content_gaps: List[ContentGap] = Field(..., description="Identified content gaps")
    category_mappings: List[CategoryMapping] = Field(..., description="Recommended category mappings")
    optimization_recommendations: List[str] = Field(..., description="Optimization recommendations")


class KnowledgeBaseOptimizer:
    """
    Knowledge base optimizer for customer support content.
    
    Provides:
    - Content analysis and categorization
    - Gap analysis and recommendations
    - Category standardization
    - Metadata enhancement suggestions
    """
    
    def __init__(self):
        """Initialize knowledge base optimizer."""
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        
        # Standard customer support categories from config
        self.target_categories = self.settings.support_categories
        
        # Category mapping rules
        self.category_mappings = {
            "FAQ": "FAQ",
            "الأسئلة الشائعة": "FAQ",
            "سياسة الخصوصية": "policies", 
            "شروط و احكام": "terms and conditions",
            "خدمات": "services",
            "insurance": "services",
            "عن الشركة": "support",
            "about": "support",
            "general": "support"
        }
        
        # Target distribution (percentage of total content)
        self.target_distribution = {
            "FAQ": 0.25,           # 25% - Frequently asked questions
            "troubleshooting": 0.20, # 20% - Technical issue resolution
            "services": 0.15,       # 15% - Service information
            "policies": 0.15,       # 15% - Privacy and policies
            "support": 0.10,        # 10% - General support
            "billing": 0.10,        # 10% - Billing and payments
            "terms and conditions": 0.05  # 5% - Terms and conditions
        }
        
        logger.info("Knowledge base optimizer initialized")
    
    async def analyze_knowledge_base(self) -> KBAnalysisResult:
        """
        Perform comprehensive knowledge base analysis.
        
        Returns:
            Analysis result with gaps and recommendations
        """
        try:
            logger.info("Starting knowledge base analysis")
            
            # Get all points from vector store
            all_points = await self._get_all_points()
            
            # Analyze current structure
            category_distribution = self._analyze_categories(all_points)
            total_documents = len(set(point.payload.get('title', 'Unknown') for point in all_points))
            
            # Generate category mappings
            category_mappings = self._generate_category_mappings(category_distribution)
            
            # Identify content gaps
            content_gaps = self._identify_content_gaps(category_distribution, len(all_points))
            
            # Generate optimization recommendations
            recommendations = self._generate_recommendations(
                category_distribution, content_gaps, len(all_points)
            )
            
            result = KBAnalysisResult(
                total_chunks=len(all_points),
                total_documents=total_documents,
                category_distribution=category_distribution,
                content_gaps=content_gaps,
                category_mappings=category_mappings,
                optimization_recommendations=recommendations
            )
            
            logger.info(
                "Knowledge base analysis completed",
                total_chunks=len(all_points),
                total_documents=total_documents,
                categories_found=len(category_distribution),
                gaps_identified=len(content_gaps)
            )
            
            return result
            
        except Exception as e:
            logger.error("Knowledge base analysis failed", error=str(e))
            raise
    
    async def _get_all_points(self) -> List[Any]:
        """Get all points from the vector store."""
        all_points = []
        offset = None
        
        while True:
            points, next_offset = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            all_points.extend(points)
            
            if next_offset is None:
                break
            offset = next_offset
        
        return all_points
    
    def _analyze_categories(self, points: List[Any]) -> Dict[str, int]:
        """Analyze category distribution in current knowledge base."""
        categories = Counter()
        
        for point in points:
            category = point.payload.get('category', 'uncategorized')
            categories[category] += 1
        
        return dict(categories)
    
    def _generate_category_mappings(self, current_categories: Dict[str, int]) -> List[CategoryMapping]:
        """Generate category mapping recommendations."""
        mappings = []
        
        for old_category, count in current_categories.items():
            if old_category in self.category_mappings:
                new_category = self.category_mappings[old_category]
                confidence = 0.9  # High confidence for predefined mappings
                reason = "Predefined mapping rule"
            else:
                # Try to infer mapping for unknown categories
                new_category, confidence, reason = self._infer_category_mapping(old_category)
            
            mappings.append(CategoryMapping(
                old_category=old_category,
                new_category=new_category,
                confidence=confidence,
                reason=reason
            ))
        
        return mappings
    
    def _infer_category_mapping(self, category: str) -> Tuple[str, float, str]:
        """Infer category mapping for unknown categories."""
        category_lower = category.lower()
        
        # Pattern-based inference
        if any(word in category_lower for word in ['faq', 'question', 'سؤال', 'أسئلة']):
            return "FAQ", 0.8, "Contains FAQ-related keywords"
        elif any(word in category_lower for word in ['policy', 'privacy', 'سياسة', 'خصوصية']):
            return "policies", 0.8, "Contains policy-related keywords"
        elif any(word in category_lower for word in ['terms', 'conditions', 'شروط', 'أحكام']):
            return "terms and conditions", 0.8, "Contains terms-related keywords"
        elif any(word in category_lower for word in ['service', 'خدمة', 'خدمات']):
            return "services", 0.8, "Contains service-related keywords"
        elif any(word in category_lower for word in ['support', 'help', 'دعم', 'مساعدة']):
            return "support", 0.8, "Contains support-related keywords"
        elif any(word in category_lower for word in ['billing', 'payment', 'فاتورة', 'دفع']):
            return "billing", 0.8, "Contains billing-related keywords"
        elif any(word in category_lower for word in ['troubleshoot', 'problem', 'issue', 'مشكلة']):
            return "troubleshooting", 0.8, "Contains troubleshooting-related keywords"
        else:
            return "support", 0.5, "Default mapping for unknown category"
    
    def _identify_content_gaps(self, current_categories: Dict[str, int], total_chunks: int) -> List[ContentGap]:
        """Identify content gaps based on target distribution."""
        gaps = []
        
        # Map current categories to target categories
        mapped_distribution = {}
        for old_cat, count in current_categories.items():
            new_cat = self.category_mappings.get(old_cat, "support")
            mapped_distribution[new_cat] = mapped_distribution.get(new_cat, 0) + count
        
        # Calculate gaps for each target category
        for target_category, target_percentage in self.target_distribution.items():
            current_count = mapped_distribution.get(target_category, 0)
            target_count = int(total_chunks * target_percentage)
            gap = max(0, target_count - current_count)
            
            # Determine priority
            if gap > 15:
                priority = "high"
            elif gap > 5:
                priority = "medium"
            else:
                priority = "low"
            
            # Generate suggested topics
            suggested_topics = self._generate_suggested_topics(target_category)
            
            gaps.append(ContentGap(
                category=target_category,
                current_count=current_count,
                target_count=target_count,
                gap=gap,
                priority=priority,
                suggested_topics=suggested_topics
            ))
        
        # Sort by gap size (highest first)
        gaps.sort(key=lambda x: x.gap, reverse=True)
        return gaps
    
    def _generate_suggested_topics(self, category: str) -> List[str]:
        """Generate suggested content topics for a category."""
        topic_suggestions = {
            "troubleshooting": [
                "Login issues and password reset",
                "App crashes and technical errors",
                "Payment processing problems",
                "Document upload failures",
                "Account access issues",
                "Browser compatibility problems"
            ],
            "billing": [
                "Payment methods and procedures",
                "Billing cycle and invoices",
                "Refund and cancellation policies",
                "Subscription management",
                "Payment failure resolution",
                "Pricing and plan information"
            ],
            "support": [
                "Contact information and hours",
                "General help and assistance",
                "Account management basics",
                "Getting started guide",
                "Feature overview",
                "Common user tasks"
            ],
            "services": [
                "Service descriptions and features",
                "Service availability and coverage",
                "Service comparison and selection",
                "Service setup and configuration",
                "Service limitations and restrictions",
                "Service updates and changes"
            ],
            "FAQ": [
                "Most common user questions",
                "Quick answers to frequent issues",
                "Step-by-step common procedures",
                "Feature explanations",
                "Account and profile questions",
                "Service-specific FAQs"
            ],
            "policies": [
                "Privacy policy details",
                "Data handling and security",
                "User rights and responsibilities",
                "Cookie and tracking policies",
                "Third-party integrations",
                "Policy updates and changes"
            ],
            "terms and conditions": [
                "Service terms and agreements",
                "User obligations and restrictions",
                "Liability and disclaimers",
                "Termination and suspension",
                "Intellectual property rights",
                "Dispute resolution procedures"
            ]
        }
        
        return topic_suggestions.get(category, [])
    
    def _generate_recommendations(
        self, 
        current_categories: Dict[str, int], 
        content_gaps: List[ContentGap],
        total_chunks: int
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Category standardization
        non_standard_categories = [cat for cat in current_categories.keys() 
                                 if cat not in self.target_categories]
        if non_standard_categories:
            recommendations.append(
                f"Standardize {len(non_standard_categories)} non-standard categories: {', '.join(non_standard_categories[:3])}{'...' if len(non_standard_categories) > 3 else ''}"
            )
        
        # Content gaps
        high_priority_gaps = [gap for gap in content_gaps if gap.priority == "high"]
        if high_priority_gaps:
            recommendations.append(
                f"Address {len(high_priority_gaps)} high-priority content gaps, starting with {high_priority_gaps[0].category} ({high_priority_gaps[0].gap} chunks needed)"
            )
        
        # Balance issues
        dominant_category = max(current_categories.items(), key=lambda x: x[1])
        if dominant_category[1] > total_chunks * 0.5:
            recommendations.append(
                f"Rebalance content - {dominant_category[0]} dominates with {dominant_category[1]} chunks ({dominant_category[1]/total_chunks*100:.1f}%)"
            )
        
        # Metadata improvements
        recommendations.append("Enhance metadata consistency across all documents")
        recommendations.append("Add intent-based tags for better query routing")
        recommendations.append("Implement content quality scoring and improvement")
        
        return recommendations


# Global instance
_kb_optimizer = None


def get_kb_optimizer() -> KnowledgeBaseOptimizer:
    """Get global knowledge base optimizer instance."""
    global _kb_optimizer
    if _kb_optimizer is None:
        _kb_optimizer = KnowledgeBaseOptimizer()
    return _kb_optimizer
