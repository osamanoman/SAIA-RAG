"""
SAIA-RAG Adaptive Retrieval System

Implements intelligent retrieval strategies based on query types, content categories,
and user context for optimal customer support responses.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .vector_store import get_vector_store
from .openai_client import get_openai_client
from .query_processor import get_query_processor
from .content_categorizer import get_content_categorizer, ContentCategory

logger = structlog.get_logger()


class RetrievalStrategy(str, Enum):
    """Different retrieval strategies for various query types."""
    STANDARD = "standard"
    FAQ_FOCUSED = "faq_focused"
    TROUBLESHOOTING = "troubleshooting"
    POLICY_LOOKUP = "policy_lookup"
    STEP_BY_STEP = "step_by_step"
    MULTI_CATEGORY = "multi_category"
    CONTEXTUAL = "contextual"


class RetrievalConfig(BaseModel):
    """Configuration for adaptive retrieval."""
    strategy: RetrievalStrategy = Field(..., description="Retrieval strategy to use")
    max_chunks: int = Field(default=8, description="Maximum chunks to retrieve")
    confidence_threshold: float = Field(default=0.35, description="Minimum confidence threshold")
    category_filters: List[str] = Field(default_factory=list, description="Category filters to apply")
    boost_factors: Dict[str, float] = Field(default_factory=dict, description="Score boost factors")
    rerank_enabled: bool = Field(default=False, description="Whether to enable reranking")
    context_expansion: bool = Field(default=False, description="Whether to expand context")


class RetrievalResult(BaseModel):
    """Enhanced retrieval result with adaptive metadata."""
    chunks: List[Dict[str, Any]] = Field(..., description="Retrieved chunks")
    strategy_used: RetrievalStrategy = Field(..., description="Strategy applied")
    total_candidates: int = Field(..., description="Total candidates considered")
    confidence_scores: List[float] = Field(..., description="Confidence scores")
    category_distribution: Dict[str, int] = Field(..., description="Category distribution")
    retrieval_metadata: Dict[str, Any] = Field(..., description="Retrieval metadata")


class AdaptiveRetriever:
    """
    Adaptive retrieval system for customer support queries.
    
    Features:
    - Query-type specific retrieval strategies
    - Category-aware filtering and boosting
    - Context-sensitive result ranking
    - Multi-strategy fallback mechanisms
    """
    
    def __init__(self):
        """Initialize adaptive retriever."""
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.openai_client = get_openai_client()
        self.query_processor = get_query_processor()
        self.content_categorizer = get_content_categorizer()
        
        # Strategy configurations
        self.strategy_configs = {
            RetrievalStrategy.FAQ_FOCUSED: RetrievalConfig(
                strategy=RetrievalStrategy.FAQ_FOCUSED,
                max_chunks=6,
                confidence_threshold=0.4,
                category_filters=["FAQ"],
                boost_factors={"FAQ": 1.3, "question_intent": 1.2},
                rerank_enabled=True
            ),
            RetrievalStrategy.TROUBLESHOOTING: RetrievalConfig(
                strategy=RetrievalStrategy.TROUBLESHOOTING,
                max_chunks=8,
                confidence_threshold=0.3,
                category_filters=["troubleshooting", "FAQ", "support"],
                boost_factors={"troubleshooting": 1.4, "procedure_intent": 1.3},
                context_expansion=True,
                rerank_enabled=True
            ),
            RetrievalStrategy.POLICY_LOOKUP: RetrievalConfig(
                strategy=RetrievalStrategy.POLICY_LOOKUP,
                max_chunks=5,
                confidence_threshold=0.45,
                category_filters=["policies", "terms and conditions"],
                boost_factors={"policies": 1.2, "terms and conditions": 1.1},
                rerank_enabled=False  # Policy content is usually precise
            ),
            RetrievalStrategy.STEP_BY_STEP: RetrievalConfig(
                strategy=RetrievalStrategy.STEP_BY_STEP,
                max_chunks=10,
                confidence_threshold=0.25,
                category_filters=["setup", "billing", "services"],
                boost_factors={"procedure_intent": 1.4, "setup": 1.3},
                context_expansion=True,
                rerank_enabled=True
            ),
            RetrievalStrategy.MULTI_CATEGORY: RetrievalConfig(
                strategy=RetrievalStrategy.MULTI_CATEGORY,
                max_chunks=12,
                confidence_threshold=0.2,
                category_filters=[],  # No filters for broad search
                boost_factors={},
                context_expansion=True,
                rerank_enabled=True
            ),
            RetrievalStrategy.STANDARD: RetrievalConfig(
                strategy=RetrievalStrategy.STANDARD,
                max_chunks=8,
                confidence_threshold=0.35,
                category_filters=[],
                boost_factors={},
                rerank_enabled=False
            )
        }
        
        logger.info("Adaptive retriever initialized")
    
    async def retrieve_adaptive(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Perform adaptive retrieval based on query analysis.
        
        Args:
            query: User query
            conversation_context: Previous conversation context
            user_context: User-specific context (preferences, history)
            
        Returns:
            Adaptive retrieval result
        """
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Analyze query to determine optimal strategy
            strategy = await self._determine_retrieval_strategy(
                query, conversation_context, user_context
            )
            
            # Step 2: Get strategy configuration
            config = self.strategy_configs[strategy]
            
            # Step 3: Execute retrieval with adaptive configuration
            retrieval_result = await self._execute_retrieval_strategy(
                query, config, conversation_context
            )
            
            # Step 4: Apply post-processing enhancements
            enhanced_result = await self._enhance_retrieval_result(
                retrieval_result, query, config
            )
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(
                "Adaptive retrieval completed",
                strategy=strategy.value,
                chunks_retrieved=len(enhanced_result.chunks),
                processing_time_ms=int(processing_time * 1000),
                avg_confidence=sum(enhanced_result.confidence_scores) / len(enhanced_result.confidence_scores) if enhanced_result.confidence_scores else 0
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error("Adaptive retrieval failed", query=query[:100], error=str(e))
            # Fallback to standard retrieval
            return await self._fallback_retrieval(query)
    
    async def _determine_retrieval_strategy(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> RetrievalStrategy:
        """Determine optimal retrieval strategy for the query."""
        
        # Process query to get classification
        enhanced_query = await self.query_processor.process_query(query)
        query_category = enhanced_query.query_type.category
        query_intent = enhanced_query.query_type.intent
        
        # Strategy decision logic
        if query_category == "faq" or query_intent == "question":
            if "how" in query.lower() or "كيف" in query.lower():
                return RetrievalStrategy.STEP_BY_STEP
            else:
                return RetrievalStrategy.FAQ_FOCUSED
        
        elif query_category == "troubleshooting" or any(word in query.lower() for word in [
            "problem", "issue", "error", "not working", "مشكلة", "خطأ", "لا يعمل"
        ]):
            return RetrievalStrategy.TROUBLESHOOTING
        
        elif query_category == "policy" or any(word in query.lower() for word in [
            "policy", "privacy", "terms", "conditions", "سياسة", "شروط", "أحكام"
        ]):
            return RetrievalStrategy.POLICY_LOOKUP
        
        elif query_category in ["setup", "billing"] or any(word in query.lower() for word in [
            "setup", "install", "configure", "payment", "billing", "إعداد", "تثبيت", "دفع"
        ]):
            return RetrievalStrategy.STEP_BY_STEP
        
        elif query_category == "general" and len(query.split()) > 10:
            # Complex queries might span multiple categories
            return RetrievalStrategy.MULTI_CATEGORY
        
        else:
            return RetrievalStrategy.STANDARD
    
    async def _execute_retrieval_strategy(
        self,
        query: str,
        config: RetrievalConfig,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Execute the specific retrieval strategy."""
        
        # Generate query embedding
        query_embedding = await self.openai_client.generate_embedding(query)
        
        # Build search filters based on strategy
        search_filters = self._build_search_filters(config)
        
        # Perform initial vector search
        search_results = self.vector_store.search_similar(
            query_vector=query_embedding,
            limit=config.max_chunks * 2,  # Get more candidates for filtering
            score_threshold=config.confidence_threshold * 0.8,  # Lower threshold for candidates
            query_filter=search_filters
        )
        
        # Apply strategy-specific processing
        processed_results = await self._apply_strategy_processing(
            search_results, query, config
        )
        
        # Apply score boosting
        boosted_results = self._apply_score_boosting(processed_results, config)
        
        # Select top results
        final_chunks = boosted_results[:config.max_chunks]
        
        # Calculate metadata
        confidence_scores = [chunk["score"] for chunk in final_chunks]
        category_distribution = self._calculate_category_distribution(final_chunks)
        
        return RetrievalResult(
            chunks=final_chunks,
            strategy_used=config.strategy,
            total_candidates=len(search_results),
            confidence_scores=confidence_scores,
            category_distribution=category_distribution,
            retrieval_metadata={
                "filters_applied": search_filters,
                "boost_factors": config.boost_factors,
                "rerank_enabled": config.rerank_enabled,
                "context_expansion": config.context_expansion
            }
        )
    
    def _build_search_filters(self, config: RetrievalConfig) -> Optional[Dict[str, Any]]:
        """Build search filters based on strategy configuration."""
        if not config.category_filters:
            return None
        
        # Build Qdrant filter for categories
        filter_conditions = []
        
        for category in config.category_filters:
            filter_conditions.append({
                "key": "category",
                "match": {"value": category}
            })
        
        if len(filter_conditions) == 1:
            return {"must": [filter_conditions[0]]}
        elif len(filter_conditions) > 1:
            return {"should": filter_conditions}
        
        return None
    
    async def _apply_strategy_processing(
        self,
        search_results: List[Dict[str, Any]],
        query: str,
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """Apply strategy-specific processing to search results."""
        
        if config.strategy == RetrievalStrategy.FAQ_FOCUSED:
            return self._process_faq_results(search_results, query)
        
        elif config.strategy == RetrievalStrategy.TROUBLESHOOTING:
            return await self._process_troubleshooting_results(search_results, query)
        
        elif config.strategy == RetrievalStrategy.POLICY_LOOKUP:
            return self._process_policy_results(search_results, query)
        
        elif config.strategy == RetrievalStrategy.STEP_BY_STEP:
            return self._process_procedural_results(search_results, query)
        
        elif config.strategy == RetrievalStrategy.MULTI_CATEGORY:
            return await self._process_multi_category_results(search_results, query)
        
        else:
            return search_results  # Standard processing
    
    def _process_faq_results(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Process results for FAQ-focused strategy."""
        processed = []
        
        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            
            # Boost FAQ content with question patterns
            boost_factor = 1.0
            if any(pattern in text.lower() for pattern in ["سؤال", "جواب", "question", "answer"]):
                boost_factor = 1.2
            
            # Boost if query keywords appear in text
            query_words = query.lower().split()
            text_lower = text.lower()
            keyword_matches = sum(1 for word in query_words if word in text_lower)
            if keyword_matches > 0:
                boost_factor += 0.1 * keyword_matches
            
            result["score"] = result["score"] * boost_factor
            processed.append(result)
        
        return sorted(processed, key=lambda x: x["score"], reverse=True)
    
    async def _process_troubleshooting_results(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Process results for troubleshooting strategy."""
        processed = []
        
        # Keywords that indicate troubleshooting content
        troubleshooting_keywords = [
            "مشكلة", "خطأ", "حل", "إصلاح", "problem", "error", "solution", "fix", "resolve"
        ]
        
        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            
            boost_factor = 1.0
            
            # Boost troubleshooting-related content
            troubleshooting_matches = sum(1 for keyword in troubleshooting_keywords if keyword in text.lower())
            if troubleshooting_matches > 0:
                boost_factor += 0.15 * troubleshooting_matches
            
            # Boost step-by-step content
            if any(pattern in text.lower() for pattern in ["خطوة", "أولاً", "step", "first"]):
                boost_factor += 0.1
            
            result["score"] = result["score"] * boost_factor
            processed.append(result)
        
        return sorted(processed, key=lambda x: x["score"], reverse=True)

    def _process_policy_results(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Process results for policy lookup strategy."""
        processed = []

        policy_keywords = [
            "سياسة", "خصوصية", "شروط", "أحكام", "policy", "privacy", "terms", "conditions"
        ]

        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            category = payload.get("category", "")

            boost_factor = 1.0

            # Strong boost for policy categories
            if category.lower() in ["policies", "terms and conditions", "سياسة الخصوصية", "شروط و احكام"]:
                boost_factor += 0.3

            # Boost policy-related keywords
            policy_matches = sum(1 for keyword in policy_keywords if keyword in text.lower())
            if policy_matches > 0:
                boost_factor += 0.1 * policy_matches

            result["score"] = result["score"] * boost_factor
            processed.append(result)

        return sorted(processed, key=lambda x: x["score"], reverse=True)

    def _process_procedural_results(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Process results for step-by-step procedural strategy."""
        processed = []

        procedural_keywords = [
            "خطوة", "أولاً", "ثانياً", "اتبع", "قم بـ", "step", "first", "then", "follow", "do"
        ]

        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")

            boost_factor = 1.0

            # Boost procedural content
            procedural_matches = sum(1 for keyword in procedural_keywords if keyword in text.lower())
            if procedural_matches > 0:
                boost_factor += 0.2 * procedural_matches

            # Boost numbered lists
            if any(pattern in text for pattern in ["1)", "2)", "3)", "١)", "٢)", "٣)"]):
                boost_factor += 0.15

            # Boost "how to" content
            if any(phrase in text.lower() for phrase in ["كيفية", "كيف", "how to", "how do"]):
                boost_factor += 0.1

            result["score"] = result["score"] * boost_factor
            processed.append(result)

        return sorted(processed, key=lambda x: x["score"], reverse=True)

    async def _process_multi_category_results(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Process results for multi-category strategy."""
        processed = []

        # Categorize query to understand what categories to boost
        enhanced_query = await self.query_processor.process_query(query)
        query_keywords = enhanced_query.query_type.keywords

        for result in search_results:
            payload = result.get("payload", {})
            text = payload.get("text", "")
            category = payload.get("category", "")

            boost_factor = 1.0

            # Boost based on keyword matches
            if query_keywords:
                keyword_matches = sum(1 for keyword in query_keywords if keyword.lower() in text.lower())
                if keyword_matches > 0:
                    boost_factor += 0.1 * keyword_matches

            # Slight boost for diverse categories to ensure variety
            category_diversity_boost = {
                "FAQ": 1.0,
                "troubleshooting": 1.1,
                "policies": 0.9,
                "services": 1.05,
                "billing": 1.1,
                "setup": 1.05
            }

            boost_factor *= category_diversity_boost.get(category, 1.0)

            result["score"] = result["score"] * boost_factor
            processed.append(result)

        return sorted(processed, key=lambda x: x["score"], reverse=True)

    def _apply_score_boosting(
        self,
        search_results: List[Dict[str, Any]],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """Apply configured score boosting factors."""
        if not config.boost_factors:
            return search_results

        boosted_results = []

        for result in search_results:
            payload = result.get("payload", {})
            category = payload.get("category", "")

            boost_factor = 1.0

            # Apply category-specific boosts
            for boost_key, boost_value in config.boost_factors.items():
                if boost_key == category:
                    boost_factor *= boost_value
                elif boost_key.endswith("_intent"):
                    # Intent-based boosting
                    intent_type = boost_key.replace("_intent", "")
                    if self._has_intent_markers(payload.get("text", ""), intent_type):
                        boost_factor *= boost_value

            result["score"] = result["score"] * boost_factor
            boosted_results.append(result)

        return sorted(boosted_results, key=lambda x: x["score"], reverse=True)

    def _has_intent_markers(self, text: str, intent_type: str) -> bool:
        """Check if text has markers for specific intent type."""
        intent_markers = {
            "question": ["؟", "?", "كيف", "ما", "هل", "how", "what", "can"],
            "procedure": ["خطوة", "أولاً", "step", "first", "follow"],
            "policy": ["سياسة", "قانون", "policy", "rule"],
            "definition": ["تعريف", "معنى", "definition", "meaning"]
        }

        markers = intent_markers.get(intent_type, [])
        return any(marker in text.lower() for marker in markers)

    async def _enhance_retrieval_result(
        self,
        result: RetrievalResult,
        query: str,
        config: RetrievalConfig
    ) -> RetrievalResult:
        """Apply post-processing enhancements to retrieval result."""

        enhanced_chunks = result.chunks

        # Apply reranking if enabled
        if config.rerank_enabled and len(enhanced_chunks) > 1:
            enhanced_chunks = await self._apply_reranking(enhanced_chunks, query)

        # Apply context expansion if enabled
        if config.context_expansion:
            enhanced_chunks = await self._expand_context(enhanced_chunks)

        # Update confidence scores after enhancements
        confidence_scores = [chunk["score"] for chunk in enhanced_chunks]

        # Recalculate category distribution
        category_distribution = self._calculate_category_distribution(enhanced_chunks)

        return RetrievalResult(
            chunks=enhanced_chunks,
            strategy_used=result.strategy_used,
            total_candidates=result.total_candidates,
            confidence_scores=confidence_scores,
            category_distribution=category_distribution,
            retrieval_metadata={
                **result.retrieval_metadata,
                "post_processing_applied": True,
                "reranking_applied": config.rerank_enabled,
                "context_expansion_applied": config.context_expansion
            }
        )

    async def _apply_reranking(
        self,
        chunks: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Apply intelligent reranking to improve relevance."""
        reranked_chunks = []

        for chunk in chunks:
            payload = chunk.get("payload", {})
            text = payload.get("text", "")

            # Calculate additional relevance signals
            relevance_score = chunk["score"]

            # Boost for exact phrase matches
            query_lower = query.lower()
            if query_lower in text.lower():
                relevance_score *= 1.2

            # Boost for title matches
            title = payload.get("title", "")
            if any(word in title.lower() for word in query.lower().split()):
                relevance_score *= 1.1

            # Update score
            chunk["score"] = relevance_score
            reranked_chunks.append(chunk)

        return sorted(reranked_chunks, key=lambda x: x["score"], reverse=True)

    async def _expand_context(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Expand context by finding related chunks."""
        # For now, return as-is. Future enhancements could include:
        # - Finding chunks from same document
        # - Finding chunks with similar tags
        # - Finding chunks from related categories
        return chunks

    def _calculate_category_distribution(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate distribution of categories in results."""
        distribution = {}

        for chunk in chunks:
            payload = chunk.get("payload", {})
            category = payload.get("category", "unknown")
            distribution[category] = distribution.get(category, 0) + 1

        return distribution

    async def _fallback_retrieval(self, query: str) -> RetrievalResult:
        """Fallback retrieval when adaptive strategy fails."""
        try:
            # Use standard strategy as fallback
            config = self.strategy_configs[RetrievalStrategy.STANDARD]
            return await self._execute_retrieval_strategy(query, config)
        except Exception as e:
            logger.error("Fallback retrieval failed", error=str(e))
            # Return empty result
            return RetrievalResult(
                chunks=[],
                strategy_used=RetrievalStrategy.STANDARD,
                total_candidates=0,
                confidence_scores=[],
                category_distribution={},
                retrieval_metadata={"fallback_used": True, "error": str(e)}
            )


# Global instance
_adaptive_retriever = None


def get_adaptive_retriever() -> AdaptiveRetriever:
    """Get global adaptive retriever instance."""
    global _adaptive_retriever
    if _adaptive_retriever is None:
        _adaptive_retriever = AdaptiveRetriever()
    return _adaptive_retriever
