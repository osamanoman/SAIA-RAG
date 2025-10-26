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
        original_query = query

        try:
            # Step 0: Reformulate with conversation context if follow-up
            logger.info(
                "Processing query with conversation context",
                has_context=bool(conversation_context),
                query_length=len(query.split()),
                query=query[:50]
            )

            if conversation_context:
                logger.info(
                    "Conversation context details",
                    context_keys=list(conversation_context.keys()) if conversation_context else [],
                    message_count=len(conversation_context.get("messages", [])) + len(conversation_context.get("recent_messages", []))
                )

                case_facts = self._extract_case_facts(conversation_context)
                logger.info(
                    "Case facts extracted",
                    has_facts=bool(case_facts),
                    fact_keys=list(case_facts.keys()) if case_facts else []
                )

                is_follow_up = self._is_follow_up_query(query)
                logger.info(
                    "Follow-up detection",
                    is_follow_up=is_follow_up,
                    query=query
                )

                if case_facts and is_follow_up:
                    query = await self._reformulate_with_context(query, case_facts)
                    preprocessing_steps.append("context_reformulation")
                    logger.info(
                        "Query reformulated with conversation context",
                        original=original_query,
                        reformulated=query
                    )
                else:
                    logger.info(
                        "Skipping reformulation",
                        has_case_facts=bool(case_facts),
                        is_follow_up=is_follow_up
                    )

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
                original_query=original_query,  # Use original, not reformulated
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

    def _extract_case_facts(self, conversation_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract case facts from conversation history.

        Args:
            conversation_context: Conversation context with message history

        Returns:
            Dictionary with case facts: plaintiff, defendant, key_issues, etc.
        """
        # Check for both "messages" and "recent_messages" keys
        messages = conversation_context.get("messages") or conversation_context.get("recent_messages") if conversation_context else None

        if not messages:
            return {}

        # Get first user message (usually contains case description)
        # Handle both "message_type" and "type" keys
        first_user_message = next(
            (msg for msg in messages if msg.get("message_type") == "user_query" or msg.get("type") == "user_query"),
            None
        )

        if not first_user_message:
            return {}

        first_query = first_user_message.get("content", "")

        # Extract entities using simple pattern matching
        case_facts = {
            "first_query": first_query,
            "query_length": len(first_query),
            "message_count": len(messages)
        }

        # Look for plaintiff/defendant patterns
        if "زوجة" in first_query or "المدعية" in first_query:
            case_facts["plaintiff_type"] = "wife"
        if "زوج" in first_query or "المدعى عليه" in first_query:
            case_facts["defendant_type"] = "husband"

        # Extract key issues
        key_issues = []
        issue_patterns = {
            "إدمان": "drug_addiction",
            "اعتداء": "assault",
            "ضرب": "physical_abuse",
            "إهمال": "neglect",
            "نفقة": "alimony",
            "حضانة": "custody",
            "فسخ": "dissolution",
            "طلاق": "divorce"
        }

        for arabic_term, english_term in issue_patterns.items():
            if arabic_term in first_query:
                key_issues.append(english_term)

        case_facts["key_issues"] = key_issues

        logger.info(
            "Extracted case facts",
            plaintiff_type=case_facts.get("plaintiff_type"),
            defendant_type=case_facts.get("defendant_type"),
            key_issues=key_issues
        )

        return case_facts

    def _is_follow_up_query(self, query: str) -> bool:
        """
        Detect if query is a follow-up (short, vague, uses pronouns).

        Examples of follow-ups:
        - "ماذا عن النفقة" (What about alimony?)
        - "والحضانة؟" (And custody?)
        - "في هذه القضية" (In this case)

        Args:
            query: User query to check

        Returns:
            True if query appears to be a follow-up
        """
        # Short queries are likely follow-ups
        if len(query.split()) <= 5:
            # Check for follow-up indicators
            follow_up_indicators = [
                "ماذا عن",  # What about
                "والحضانة",  # And custody
                "والنفقة",  # And alimony
                "في هذه القضية",  # In this case
                "في القضية",  # In the case
                "أيضا",  # Also
                "كذلك",  # As well
                "بالنسبة",  # Regarding
                "وماذا",  # And what
            ]

            for indicator in follow_up_indicators:
                if indicator in query:
                    logger.info("Detected follow-up query", query=query, indicator=indicator)
                    return True

        return False

    async def _reformulate_with_context(
        self,
        query: str,
        case_facts: Dict[str, Any]
    ) -> str:
        """
        Reformulate follow-up query with case context.

        Examples:
        - "ماذا عن النفقة" → "ما هي حقوق الزوجة في النفقة في قضية إدمان الزوج وإهماله؟"
        - "What about custody?" → "What are the wife's custody rights given husband's drug addiction?"

        Args:
            query: Original follow-up query
            case_facts: Extracted case facts from conversation history

        Returns:
            Reformulated query with case context
        """
        try:
            # Build context string from case facts
            context_parts = []

            if case_facts.get("plaintiff_type"):
                context_parts.append(f"Plaintiff: {case_facts['plaintiff_type']}")

            if case_facts.get("defendant_type"):
                context_parts.append(f"Defendant: {case_facts['defendant_type']}")

            if case_facts.get("key_issues"):
                context_parts.append(f"Key issues: {', '.join(case_facts['key_issues'])}")

            context_string = "; ".join(context_parts)

            # Use LLM to reformulate
            reformulation_prompt = f"""You are a legal query reformulation assistant.

Original follow-up query: {query}

Case context: {context_string}

Original case description: {case_facts.get('first_query', '')[:500]}

Task: Reformulate the follow-up query to be specific and include relevant case context.
The reformulated query should be a complete, standalone question that includes:
1. The specific legal topic from the follow-up (e.g., alimony, custody)
2. Relevant case facts (e.g., drug addiction, neglect)
3. The parties involved (e.g., wife, husband)

Respond with ONLY the reformulated query in the same language as the original query.
Do not add explanations or additional text."""

            messages = [{"role": "user", "content": reformulation_prompt}]

            result = await self.openai_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=200
            )

            reformulated = result["content"].strip()

            logger.info(
                "Query reformulated with context",
                original=query,
                reformulated=reformulated,
                context_used=bool(context_string)
            )

            return reformulated

        except Exception as e:
            logger.error("Query reformulation failed", error=str(e))
            return query  # Fallback to original query


# Global instance
_query_processor = None


def get_query_processor() -> QueryProcessor:
    """Get global query processor instance."""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor
