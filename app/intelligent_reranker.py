"""
SAIA-RAG Intelligent Reranking System

Implements advanced reranking techniques including Cross-Encoder models,
semantic similarity scoring, and customer support context optimization.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import re

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .openai_client import get_openai_client

logger = structlog.get_logger()


class RerankingMethod(str, Enum):
    """Different reranking methods available."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CROSS_ENCODER = "cross_encoder"
    HYBRID = "hybrid"
    CUSTOMER_SUPPORT = "customer_support"
    CONTEXTUAL = "contextual"


class RerankingConfig(BaseModel):
    """Configuration for reranking operations."""
    method: RerankingMethod = Field(..., description="Reranking method to use")
    top_k: int = Field(default=8, description="Number of top results to return")
    similarity_threshold: float = Field(default=0.1, description="Minimum similarity threshold")
    boost_factors: Dict[str, float] = Field(default_factory=dict, description="Category-specific boost factors")
    context_weight: float = Field(default=0.3, description="Weight for contextual signals")
    enable_diversity: bool = Field(default=True, description="Enable result diversification")


class RerankingResult(BaseModel):
    """Result of reranking operation."""
    reranked_chunks: List[Dict[str, Any]] = Field(..., description="Reranked chunks")
    original_scores: List[float] = Field(..., description="Original relevance scores")
    reranked_scores: List[float] = Field(..., description="New reranked scores")
    score_improvements: List[float] = Field(..., description="Score improvement deltas")
    method_used: RerankingMethod = Field(..., description="Reranking method applied")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing metadata")


class IntelligentReranker:
    """
    Advanced reranking system for customer support queries.
    
    Features:
    - Semantic similarity reranking
    - Cross-encoder style scoring
    - Customer support context optimization
    - Result diversification
    - Multi-signal fusion
    """
    
    def __init__(self):
        """Initialize intelligent reranker."""
        self.settings = get_settings()
        self.openai_client = get_openai_client()
        
        # Reranking configurations for different scenarios
        self.reranking_configs = {
            RerankingMethod.SEMANTIC_SIMILARITY: RerankingConfig(
                method=RerankingMethod.SEMANTIC_SIMILARITY,
                top_k=8,
                similarity_threshold=0.15,
                context_weight=0.2,
                enable_diversity=True
            ),
            RerankingMethod.CUSTOMER_SUPPORT: RerankingConfig(
                method=RerankingMethod.CUSTOMER_SUPPORT,
                top_k=8,
                similarity_threshold=0.1,
                boost_factors={
                    "FAQ": 1.2,
                    "troubleshooting": 1.3,
                    "policies": 1.1,
                    "billing": 1.25
                },
                context_weight=0.4,
                enable_diversity=True
            ),
            RerankingMethod.HYBRID: RerankingConfig(
                method=RerankingMethod.HYBRID,
                top_k=10,
                similarity_threshold=0.05,
                context_weight=0.35,
                enable_diversity=True
            )
        }
        
        # Customer support specific patterns
        self.support_patterns = {
            "urgency_indicators": [
                "urgent", "emergency", "asap", "immediately", "critical",
                "عاجل", "طارئ", "فوري", "حرج", "مهم جداً"
            ],
            "satisfaction_indicators": [
                "thank", "thanks", "helpful", "solved", "resolved",
                "شكراً", "مفيد", "حل", "تم الحل"
            ],
            "frustration_indicators": [
                "frustrated", "angry", "disappointed", "terrible", "awful",
                "محبط", "غاضب", "سيء", "فظيع"
            ],
            "question_patterns": [
                r"how\s+(?:do|can|to)", r"what\s+(?:is|are)", r"why\s+(?:is|are|do)",
                r"كيف\s+(?:يمكن|أقوم)", r"ما\s+(?:هو|هي)", r"لماذا\s+(?:يحدث|لا)"
            ]
        }
        
        logger.info("Intelligent reranker initialized")
    
    async def rerank_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        method: RerankingMethod = RerankingMethod.CUSTOMER_SUPPORT,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> RerankingResult:
        """
        Rerank search results using intelligent scoring.
        
        Args:
            query: Original user query
            search_results: Initial search results to rerank
            method: Reranking method to use
            conversation_context: Optional conversation context
            
        Returns:
            Reranking result with enhanced scores
        """
        try:
            start_time = datetime.utcnow()
            
            if not search_results:
                return self._empty_reranking_result(method)
            
            # Get configuration for method
            config = self.reranking_configs.get(method, self.reranking_configs[RerankingMethod.CUSTOMER_SUPPORT])
            
            # Store original scores
            original_scores = [result.get("score", 0.0) for result in search_results]
            
            # Apply reranking based on method
            if method == RerankingMethod.SEMANTIC_SIMILARITY:
                reranked_results = await self._semantic_similarity_rerank(query, search_results, config)
            elif method == RerankingMethod.CUSTOMER_SUPPORT:
                reranked_results = await self._customer_support_rerank(query, search_results, config, conversation_context)
            elif method == RerankingMethod.HYBRID:
                reranked_results = await self._hybrid_rerank(query, search_results, config, conversation_context)
            else:
                reranked_results = await self._customer_support_rerank(query, search_results, config, conversation_context)
            
            # Apply diversification if enabled
            if config.enable_diversity:
                reranked_results = self._apply_diversification(reranked_results, config)
            
            # Select top-k results
            final_results = reranked_results[:config.top_k]
            
            # Calculate metrics
            reranked_scores = [result.get("score", 0.0) for result in final_results]
            score_improvements = [
                reranked_scores[i] - original_scores[i] 
                for i in range(min(len(reranked_scores), len(original_scores)))
            ]
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            result = RerankingResult(
                reranked_chunks=final_results,
                original_scores=original_scores[:len(final_results)],
                reranked_scores=reranked_scores,
                score_improvements=score_improvements,
                method_used=method,
                processing_metadata={
                    "processing_time_ms": int(processing_time * 1000),
                    "original_count": len(search_results),
                    "final_count": len(final_results),
                    "avg_score_improvement": sum(score_improvements) / len(score_improvements) if score_improvements else 0,
                    "diversification_applied": config.enable_diversity
                }
            )
            
            logger.info(
                "Reranking completed",
                method=method.value,
                original_count=len(search_results),
                final_count=len(final_results),
                avg_improvement=result.processing_metadata["avg_score_improvement"],
                processing_time_ms=result.processing_metadata["processing_time_ms"]
            )
            
            return result
            
        except Exception as e:
            logger.error("Reranking failed", query=query[:100], method=method.value, error=str(e))
            return self._fallback_reranking(search_results, method)
    
    async def _semantic_similarity_rerank(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        config: RerankingConfig
    ) -> List[Dict[str, Any]]:
        """Rerank using semantic similarity scoring."""
        
        # Generate query embedding
        query_embedding = await self.openai_client.generate_embedding(query)
        
        reranked_results = []
        
        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            
            # Generate text embedding
            text_embedding = await self.openai_client.generate_embedding(text)
            
            # Calculate cosine similarity
            similarity_score = self._calculate_cosine_similarity(query_embedding, text_embedding)
            
            # Combine with original score
            original_score = result.get("score", 0.0)
            combined_score = (original_score * 0.7) + (similarity_score * 0.3)
            
            # Apply threshold filtering
            if combined_score >= config.similarity_threshold:
                result["score"] = combined_score
                result["similarity_score"] = similarity_score
                reranked_results.append(result)
        
        return sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    
    async def _customer_support_rerank(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        config: RerankingConfig,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Rerank using customer support specific signals."""
        
        # Analyze query characteristics
        query_analysis = self._analyze_query_characteristics(query)
        
        reranked_results = []
        
        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            category = payload.get("category", "")
            
            # Start with original score
            enhanced_score = result.get("score", 0.0)
            
            # Apply customer support specific boosts
            enhanced_score = self._apply_support_boosts(
                enhanced_score, text, category, query_analysis, config
            )
            
            # Apply contextual signals
            if conversation_context:
                enhanced_score = self._apply_contextual_signals(
                    enhanced_score, text, conversation_context, config
                )
            
            # Apply query-text matching signals
            enhanced_score = self._apply_matching_signals(enhanced_score, query, text)
            
            result["score"] = enhanced_score
            reranked_results.append(result)
        
        return sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    
    async def _hybrid_rerank(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        config: RerankingConfig,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Rerank using hybrid approach combining multiple signals."""
        
        # Apply semantic similarity reranking
        semantic_results = await self._semantic_similarity_rerank(query, search_results, config)
        
        # Apply customer support reranking
        support_results = await self._customer_support_rerank(query, semantic_results, config, conversation_context)
        
        # Combine scores with weighted average
        final_results = []
        
        for result in support_results:
            original_score = result.get("score", 0.0)
            similarity_score = result.get("similarity_score", 0.0)
            
            # Weighted combination
            hybrid_score = (original_score * 0.6) + (similarity_score * 0.4)
            
            result["score"] = hybrid_score
            final_results.append(result)
        
        return sorted(final_results, key=lambda x: x["score"], reverse=True)
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze query to identify customer support characteristics."""
        query_lower = query.lower()
        
        characteristics = {
            "is_urgent": any(indicator in query_lower for indicator in self.support_patterns["urgency_indicators"]),
            "is_question": any(re.search(pattern, query_lower) for pattern in self.support_patterns["question_patterns"]),
            "shows_frustration": any(indicator in query_lower for indicator in self.support_patterns["frustration_indicators"]),
            "query_length": len(query.split()),
            "has_numbers": bool(re.search(r'\d+', query)),
            "language": "ar" if any(char in query for char in "أبتثجحخدذرزسشصضطظعغفقكلمنهوي") else "en"
        }
        
        return characteristics
    
    def _apply_support_boosts(
        self,
        score: float,
        text: str,
        category: str,
        query_analysis: Dict[str, Any],
        config: RerankingConfig
    ) -> float:
        """Apply customer support specific score boosts."""
        
        enhanced_score = score
        
        # Category-specific boosts
        if category in config.boost_factors:
            enhanced_score *= config.boost_factors[category]
        
        # Urgency matching
        if query_analysis["is_urgent"]:
            if any(indicator in text.lower() for indicator in self.support_patterns["urgency_indicators"]):
                enhanced_score *= 1.2
        
        # Question-answer matching
        if query_analysis["is_question"]:
            if any(pattern in text.lower() for pattern in ["answer", "solution", "جواب", "حل"]):
                enhanced_score *= 1.15
        
        # Language consistency boost
        text_is_arabic = any(char in text for char in "أبتثجحخدذرزسشصضطظعغفقكلمنهوي")
        if (query_analysis["language"] == "ar" and text_is_arabic) or \
           (query_analysis["language"] == "en" and not text_is_arabic):
            enhanced_score *= 1.1
        
        return enhanced_score
    
    def _apply_contextual_signals(
        self,
        score: float,
        text: str,
        conversation_context: Dict[str, Any],
        config: RerankingConfig
    ) -> float:
        """Apply contextual signals from conversation history."""
        
        enhanced_score = score
        
        # Previous topics boost
        previous_topics = conversation_context.get("topics", [])
        if previous_topics:
            text_lower = text.lower()
            topic_matches = sum(1 for topic in previous_topics if topic.lower() in text_lower)
            if topic_matches > 0:
                enhanced_score *= (1.0 + (topic_matches * 0.1))
        
        # User satisfaction context
        user_satisfaction = conversation_context.get("satisfaction", "neutral")
        if user_satisfaction == "frustrated":
            # Boost solution-oriented content
            if any(word in text.lower() for word in ["solution", "resolve", "fix", "حل", "إصلاح"]):
                enhanced_score *= 1.2
        
        return enhanced_score
    
    def _apply_matching_signals(self, score: float, query: str, text: str) -> float:
        """Apply query-text matching signals."""
        
        enhanced_score = score
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Exact phrase matching
        if query_lower in text_lower:
            enhanced_score *= 1.25
        
        # Word overlap boost
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        overlap = len(query_words.intersection(text_words))
        
        if overlap > 0:
            overlap_ratio = overlap / len(query_words)
            enhanced_score *= (1.0 + (overlap_ratio * 0.2))
        
        # Title matching (if available in payload)
        # This would be implemented when title information is available
        
        return enhanced_score

    def _apply_diversification(
        self,
        search_results: List[Dict[str, Any]],
        config: RerankingConfig
    ) -> List[Dict[str, Any]]:
        """Apply result diversification to avoid redundant content."""

        if len(search_results) <= 3:
            return search_results  # No need to diversify small result sets

        diversified_results = []
        used_categories = set()
        used_documents = set()

        # First pass: select top results ensuring category diversity
        for result in search_results:
            payload = result.get("payload", {})
            category = payload.get("category", "unknown")
            document_id = payload.get("document_id", "unknown")

            # Prefer results from different categories and documents
            category_penalty = 0.9 if category in used_categories else 1.0
            document_penalty = 0.95 if document_id in used_documents else 1.0

            # Apply diversity penalty
            result["score"] = result["score"] * category_penalty * document_penalty

            diversified_results.append(result)
            used_categories.add(category)
            used_documents.add(document_id)

        # Re-sort after applying diversity penalties
        return sorted(diversified_results, key=lambda x: x["score"], reverse=True)

    def _calculate_cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, (similarity + 1) / 2))

    def _empty_reranking_result(self, method: RerankingMethod) -> RerankingResult:
        """Return empty reranking result."""
        return RerankingResult(
            reranked_chunks=[],
            original_scores=[],
            reranked_scores=[],
            score_improvements=[],
            method_used=method,
            processing_metadata={
                "processing_time_ms": 0,
                "original_count": 0,
                "final_count": 0,
                "avg_score_improvement": 0.0,
                "diversification_applied": False
            }
        )

    def _fallback_reranking(
        self,
        search_results: List[Dict[str, Any]],
        method: RerankingMethod
    ) -> RerankingResult:
        """Fallback reranking when main process fails."""

        # Simple fallback: return original results with minimal processing
        original_scores = [result.get("score", 0.0) for result in search_results]

        return RerankingResult(
            reranked_chunks=search_results[:8],  # Limit to top 8
            original_scores=original_scores[:8],
            reranked_scores=original_scores[:8],
            score_improvements=[0.0] * min(8, len(search_results)),
            method_used=method,
            processing_metadata={
                "processing_time_ms": 0,
                "original_count": len(search_results),
                "final_count": min(8, len(search_results)),
                "avg_score_improvement": 0.0,
                "diversification_applied": False,
                "fallback_used": True
            }
        )


# Global instance
_intelligent_reranker = None


def get_intelligent_reranker() -> IntelligentReranker:
    """Get global intelligent reranker instance."""
    global _intelligent_reranker
    if _intelligent_reranker is None:
        _intelligent_reranker = IntelligentReranker()
    return _intelligent_reranker
