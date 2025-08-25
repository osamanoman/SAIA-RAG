"""
SAIA-RAG Query Processing Module

Implements query enhancement and preprocessing for improved customer support retrieval.
Follows best practices from RAG research for query optimization.
"""

import re
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .openai_client import get_openai_client

logger = structlog.get_logger()


class QueryType(BaseModel):
    """Query classification result."""
    category: str = Field(..., description="Query category (troubleshooting, billing, setup, general, policies)")
    confidence: float = Field(..., description="Classification confidence (0.0-1.0)")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    intent: str = Field(..., description="User intent (question, complaint, request, etc.)")


class EnhancedQuery(BaseModel):
    """Enhanced query result."""
    original_query: str = Field(..., description="Original user query")
    enhanced_query: str = Field(..., description="Enhanced query for better retrieval")
    query_type: QueryType = Field(..., description="Query classification")
    preprocessing_applied: List[str] = Field(default_factory=list, description="List of preprocessing steps applied")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class QueryProcessor:
    """
    Advanced query processing for customer support RAG.
    
    Implements:
    - Query cleaning and normalization
    - Query type classification
    - Query enhancement for better retrieval
    - Support-specific query transformations
    """
    
    def __init__(self):
        """Initialize query processor with dependencies."""
        self.settings = get_settings()
        self.openai_client = get_openai_client()
        
        # Customer support specific patterns
        self.error_patterns = [
            r"error\s*\d+", r"not working", r"broken", r"failed", r"issue", r"problem"
        ]
        self.billing_patterns = [
            r"bill", r"payment", r"charge", r"cost", r"price", r"refund", r"subscription"
        ]
        self.setup_patterns = [
            r"how to", r"setup", r"install", r"configure", r"getting started", r"first time"
        ]
        
        logger.info("Query processor initialized")
    
    async def process_query(
        self,
        query: str,
        channel: str = "default",
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> EnhancedQuery:
        """
        Process and enhance a user query for better retrieval.
        
        Args:
            query: Original user query
            channel: Channel type (whatsapp, chat, email, etc.)
            conversation_context: Optional conversation context
            
        Returns:
            Enhanced query with metadata
        """
        start_time = datetime.utcnow()
        preprocessing_steps = []
        
        try:
            # Step 1: Clean and normalize query
            cleaned_query = self._clean_query(query)
            if cleaned_query != query:
                preprocessing_steps.append("cleaning")
            
            # Step 2: Classify query type
            query_type = await self._classify_query(cleaned_query)
            preprocessing_steps.append("classification")
            
            # Step 3: Apply channel-specific preprocessing
            channel_enhanced = self._apply_channel_preprocessing(cleaned_query, channel)
            if channel_enhanced != cleaned_query:
                preprocessing_steps.append("channel_optimization")
            
            # Step 4: Apply support-specific enhancements
            enhanced_query = await self._enhance_for_support(channel_enhanced, query_type)
            if enhanced_query != channel_enhanced:
                preprocessing_steps.append("support_enhancement")
            
            # Step 5: Apply query expansion if enabled
            if self.settings.enable_query_enhancement:
                expanded_query = await self._expand_query(enhanced_query, query_type)
                if expanded_query != enhanced_query:
                    enhanced_query = expanded_query
                    preprocessing_steps.append("query_expansion")
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = EnhancedQuery(
                original_query=query,
                enhanced_query=enhanced_query,
                query_type=query_type,
                preprocessing_applied=preprocessing_steps,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(
                "Query processed successfully",
                original_length=len(query),
                enhanced_length=len(enhanced_query),
                category=query_type.category,
                steps_applied=preprocessing_steps,
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error("Query processing failed", query=query[:100], error=str(e))
            # Return minimal enhancement on failure
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return EnhancedQuery(
                original_query=query,
                enhanced_query=query,  # Fallback to original
                query_type=QueryType(
                    category="general",
                    confidence=0.5,
                    keywords=[],
                    intent="question"
                ),
                preprocessing_applied=["error_fallback"],
                processing_time_ms=processing_time_ms
            )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Fix common typos and abbreviations
        replacements = {
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\bpls\b': 'please',
            r'\bthx\b': 'thanks',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\bdoesnt\b': 'does not',
            r'\bcant\b': 'cannot',
            r'\bwont\b': 'will not',
        }
        
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    async def _classify_query(self, query: str) -> QueryType:
        """Classify the query into support categories."""
        query_lower = query.lower()
        
        # Pattern-based classification
        if any(re.search(pattern, query_lower) for pattern in self.error_patterns):
            category = "troubleshooting"
            confidence = 0.8
        elif any(re.search(pattern, query_lower) for pattern in self.billing_patterns):
            category = "billing"
            confidence = 0.8
        elif any(re.search(pattern, query_lower) for pattern in self.setup_patterns):
            category = "setup"
            confidence = 0.8
        else:
            category = "general"
            confidence = 0.6
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Determine intent
        intent = self._determine_intent(query)
        
        return QueryType(
            category=category,
            confidence=confidence,
            keywords=keywords,
            intent=intent
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'can', 'could', 'should', 'would', 'will'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _determine_intent(self, query: str) -> str:
        """Determine user intent from the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'what', 'where', 'when', 'why', 'which']):
            return "question"
        elif any(word in query_lower for word in ['help', 'support', 'assist', 'need']):
            return "request"
        elif any(word in query_lower for word in ['problem', 'issue', 'error', 'broken', 'not working']):
            return "complaint"
        elif any(word in query_lower for word in ['thank', 'thanks', 'appreciate']):
            return "gratitude"
        else:
            return "general"
    
    def _apply_channel_preprocessing(self, query: str, channel: str) -> str:
        """Apply channel-specific preprocessing."""
        # Removed WhatsApp-specific preprocessing to ensure consistent responses
        # across all channels (WhatsApp, web UI, etc.)
        return query
    
    async def _enhance_for_support(self, query: str, query_type: QueryType) -> str:
        """Enhance query specifically for customer support context."""
        # Add category context to improve retrieval
        category_context = {
            "troubleshooting": "technical issue help",
            "billing": "payment and billing question",
            "setup": "installation and setup guide",
            "general": "customer support",
            "policies": "policy and terms information"
        }
        
        context = category_context.get(query_type.category, "customer support")
        
        # For very short queries, add more context
        if len(query.split()) < 3:
            return f"{context}: {query}"
        
        return query
    
    async def _expand_query(self, query: str, query_type: QueryType) -> str:
        """Expand query using LLM for better retrieval."""
        try:
            expansion_prompt = f"""
            Rewrite this customer support query to be more specific and effective for knowledge retrieval.
            
            Original query: "{query}"
            Category: {query_type.category}
            Intent: {query_type.intent}
            
            Guidelines:
            - Keep the core meaning intact
            - Add relevant technical terms if applicable
            - Make it more specific for better document matching
            - Keep it concise (max 2x original length)
            
            Enhanced query:
            """
            
            messages = [{"role": "user", "content": expansion_prompt}]
            
            result = await self.openai_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            
            enhanced = result["content"].strip()
            
            # Validate enhancement (don't make it too long)
            if len(enhanced) <= len(query) * 2.5:
                return enhanced
            else:
                return query  # Fallback to original if too long
                
        except Exception as e:
            logger.warning("Query expansion failed", error=str(e))
            return query  # Fallback to original


# Global instance
_query_processor = None


def get_query_processor() -> QueryProcessor:
    """Get global query processor instance."""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor
