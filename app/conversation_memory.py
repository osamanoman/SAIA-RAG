"""
SAIA-RAG Conversation Memory and Context Management

Implements intelligent conversation memory for multi-turn interactions,
tracking conversation history, user preferences, and session state.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

import structlog
from pydantic import BaseModel, Field

from .config import get_settings

logger = structlog.get_logger()


class ConversationState(str, Enum):
    """Conversation state types."""
    ACTIVE = "active"
    IDLE = "idle"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


class MessageType(str, Enum):
    """Message types in conversation."""
    USER_QUERY = "user_query"
    AI_RESPONSE = "ai_response"
    SYSTEM_MESSAGE = "system_message"
    ESCALATION = "escalation"


class UserSentiment(str, Enum):
    """User sentiment states."""
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    CONFUSED = "confused"


class ConversationMessage(BaseModel):
    """Individual message in conversation."""
    message_id: str = Field(..., description="Unique message identifier")
    timestamp: datetime = Field(..., description="Message timestamp")
    message_type: MessageType = Field(..., description="Type of message")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    confidence_score: Optional[float] = Field(None, description="AI response confidence")
    sources: List[str] = Field(default_factory=list, description="Knowledge sources used")


class ConversationContext(BaseModel):
    """Conversation context and state."""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    state: ConversationState = Field(default=ConversationState.ACTIVE, description="Conversation state")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation start time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    # Conversation content
    messages: List[ConversationMessage] = Field(default_factory=list, description="Conversation messages")
    
    # Context tracking
    topics: List[str] = Field(default_factory=list, description="Discussed topics")
    categories: List[str] = Field(default_factory=list, description="Content categories accessed")
    user_sentiment: UserSentiment = Field(default=UserSentiment.NEUTRAL, description="Current user sentiment")
    
    # User preferences and behavior
    language_preference: str = Field(default="ar", description="User language preference")
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict, description="User interaction patterns")
    
    # Session metadata
    total_messages: int = Field(default=0, description="Total message count")
    resolution_attempts: int = Field(default=0, description="Number of resolution attempts")
    escalation_triggers: List[str] = Field(default_factory=list, description="Escalation trigger events")


class ConversationSummary(BaseModel):
    """Conversation summary for context compression."""
    conversation_id: str = Field(..., description="Conversation identifier")
    summary_text: str = Field(..., description="Conversation summary")
    key_topics: List[str] = Field(..., description="Key topics discussed")
    resolution_status: str = Field(..., description="Resolution status")
    user_satisfaction: str = Field(..., description="User satisfaction level")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Summary creation time")


class ConversationMemoryManager:
    """
    Conversation memory and context management system.
    
    Features:
    - Multi-turn conversation tracking
    - Context-aware response generation
    - User sentiment analysis
    - Topic and category tracking
    - Session state management
    - Conversation summarization
    """
    
    def __init__(self):
        """Initialize conversation memory manager."""
        self.settings = get_settings()
        
        # In-memory storage for active conversations
        # In production, this would be backed by Redis or database
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_summaries: Dict[str, ConversationSummary] = {}
        
        # Configuration
        self.max_conversation_length = 50  # Maximum messages per conversation
        self.context_window_size = 10      # Messages to include in context
        self.idle_timeout_minutes = 30     # Minutes before marking conversation idle
        self.summary_trigger_length = 20   # Messages before creating summary
        
        # Sentiment analysis patterns
        self.sentiment_patterns = {
            UserSentiment.SATISFIED: [
                "thank", "thanks", "helpful", "solved", "resolved", "perfect", "great",
                "شكراً", "مفيد", "حل", "تم الحل", "ممتاز", "رائع"
            ],
            UserSentiment.FRUSTRATED: [
                "frustrated", "annoying", "difficult", "complicated", "confusing",
                "محبط", "مزعج", "صعب", "معقد", "محير"
            ],
            UserSentiment.ANGRY: [
                "angry", "furious", "terrible", "awful", "worst", "hate",
                "غاضب", "سيء", "فظيع", "أسوأ"
            ],
            UserSentiment.CONFUSED: [
                "confused", "don't understand", "unclear", "what", "how",
                "محير", "لا أفهم", "غير واضح", "ماذا", "كيف"
            ]
        }
        
        logger.info("Conversation memory manager initialized")
    
    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """
        Start a new conversation.
        
        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier
            initial_context: Optional initial context
            
        Returns:
            New conversation context
        """
        try:
            # Generate conversation ID
            conversation_id = self._generate_conversation_id(user_id, session_id)
            
            # Create conversation context
            conversation = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id or self._generate_session_id(),
                state=ConversationState.ACTIVE
            )
            
            # Apply initial context if provided
            if initial_context:
                conversation.language_preference = initial_context.get("language", "ar")
                conversation.interaction_patterns = initial_context.get("patterns", {})
            
            # Store conversation
            self.active_conversations[conversation_id] = conversation
            
            logger.info(
                "Conversation started",
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=conversation.session_id
            )
            
            return conversation
            
        except Exception as e:
            logger.error("Failed to start conversation", error=str(e))
            raise
    
    async def add_message(
        self,
        conversation_id: str,
        content: str,
        message_type: MessageType,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        sources: Optional[List[str]] = None
    ) -> ConversationMessage:
        """
        Add a message to conversation.
        
        Args:
            conversation_id: Conversation identifier
            content: Message content
            message_type: Type of message
            metadata: Optional message metadata
            confidence_score: Optional AI confidence score
            sources: Optional knowledge sources
            
        Returns:
            Created message
        """
        try:
            # Get conversation
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Create message
            message = ConversationMessage(
                message_id=self._generate_message_id(conversation_id),
                timestamp=datetime.utcnow(),
                message_type=message_type,
                content=content,
                metadata=metadata or {},
                confidence_score=confidence_score,
                sources=sources or []
            )
            
            # Add to conversation
            conversation.messages.append(message)
            conversation.total_messages += 1
            conversation.updated_at = datetime.utcnow()
            
            # Update conversation context
            await self._update_conversation_context(conversation, message)
            
            # Check if conversation needs summarization
            if len(conversation.messages) >= self.summary_trigger_length:
                await self._create_conversation_summary(conversation)
            
            logger.info(
                "Message added to conversation",
                conversation_id=conversation_id,
                message_type=message_type.value,
                total_messages=conversation.total_messages
            )
            
            return message
            
        except Exception as e:
            logger.error("Failed to add message", conversation_id=conversation_id, error=str(e))
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation by ID."""
        return self.active_conversations.get(conversation_id)
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        include_messages: bool = True,
        context_window: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation context for AI processing.
        
        Args:
            conversation_id: Conversation identifier
            include_messages: Whether to include message history
            context_window: Number of recent messages to include
            
        Returns:
            Conversation context dictionary
        """
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                return None
            
            context = {
                "conversation_id": conversation_id,
                "state": conversation.state.value,
                "user_sentiment": conversation.user_sentiment.value,
                "language_preference": conversation.language_preference,
                "topics": conversation.topics,
                "categories": conversation.categories,
                "total_messages": conversation.total_messages,
                "resolution_attempts": conversation.resolution_attempts,
                "session_duration_minutes": self._calculate_session_duration(conversation)
            }
            
            # Include recent messages if requested
            if include_messages:
                window_size = context_window or self.context_window_size
                recent_messages = conversation.messages[-window_size:] if conversation.messages else []
                
                context["recent_messages"] = [
                    {
                        "type": msg.message_type.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "confidence": msg.confidence_score
                    }
                    for msg in recent_messages
                ]
            
            # Include conversation summary if available
            if conversation_id in self.conversation_summaries:
                summary = self.conversation_summaries[conversation_id]
                context["summary"] = {
                    "text": summary.summary_text,
                    "key_topics": summary.key_topics,
                    "resolution_status": summary.resolution_status
                }
            
            return context
            
        except Exception as e:
            logger.error("Failed to get conversation context", conversation_id=conversation_id, error=str(e))
            return None
    
    async def update_user_sentiment(
        self,
        conversation_id: str,
        message_content: str
    ) -> UserSentiment:
        """Update user sentiment based on message content."""
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                return UserSentiment.NEUTRAL
            
            # Analyze sentiment from message content
            detected_sentiment = self._analyze_sentiment(message_content)
            
            # Update conversation sentiment
            conversation.user_sentiment = detected_sentiment
            conversation.updated_at = datetime.utcnow()
            
            logger.info(
                "User sentiment updated",
                conversation_id=conversation_id,
                sentiment=detected_sentiment.value
            )
            
            return detected_sentiment
            
        except Exception as e:
            logger.error("Failed to update sentiment", conversation_id=conversation_id, error=str(e))
            return UserSentiment.NEUTRAL
    
    async def check_escalation_triggers(
        self,
        conversation_id: str,
        message_content: str,
        confidence_score: Optional[float] = None
    ) -> bool:
        """
        Check if conversation should be escalated to human.
        
        Args:
            conversation_id: Conversation identifier
            message_content: Latest message content
            confidence_score: AI response confidence
            
        Returns:
            True if escalation is recommended
        """
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                return False
            
            escalation_triggers = []
            
            # Check sentiment-based triggers
            if conversation.user_sentiment in [UserSentiment.ANGRY, UserSentiment.FRUSTRATED]:
                escalation_triggers.append("negative_sentiment")
            
            # Check confidence-based triggers
            if confidence_score is not None and confidence_score < 0.3:
                escalation_triggers.append("low_confidence")
            
            # Check repetition-based triggers
            if conversation.resolution_attempts >= 3:
                escalation_triggers.append("multiple_resolution_attempts")
            
            # Check conversation length triggers
            if conversation.total_messages >= 20:
                escalation_triggers.append("long_conversation")
            
            # Check for explicit escalation requests
            escalation_keywords = [
                "human", "person", "agent", "representative", "manager",
                "إنسان", "شخص", "وكيل", "ممثل", "مدير"
            ]
            if any(keyword in message_content.lower() for keyword in escalation_keywords):
                escalation_triggers.append("explicit_request")
            
            # Update escalation triggers
            conversation.escalation_triggers.extend(escalation_triggers)
            
            # Determine if escalation is needed
            should_escalate = len(escalation_triggers) > 0
            
            if should_escalate:
                logger.info(
                    "Escalation triggers detected",
                    conversation_id=conversation_id,
                    triggers=escalation_triggers
                )
            
            return should_escalate
            
        except Exception as e:
            logger.error("Failed to check escalation triggers", conversation_id=conversation_id, error=str(e))
            return False

    async def _update_conversation_context(
        self,
        conversation: ConversationContext,
        message: ConversationMessage
    ) -> None:
        """Update conversation context based on new message."""
        try:
            # Extract topics from message content
            if message.message_type == MessageType.USER_QUERY:
                topics = self._extract_topics(message.content)
                for topic in topics:
                    if topic not in conversation.topics:
                        conversation.topics.append(topic)

                # Update user sentiment
                conversation.user_sentiment = self._analyze_sentiment(message.content)

            # Track categories from AI responses
            elif message.message_type == MessageType.AI_RESPONSE:
                if message.sources:
                    for source in message.sources:
                        # Extract category from source metadata if available
                        category = self._extract_category_from_source(source)
                        if category and category not in conversation.categories:
                            conversation.categories.append(category)

                # Increment resolution attempts for AI responses
                conversation.resolution_attempts += 1

            # Update interaction patterns
            self._update_interaction_patterns(conversation, message)

        except Exception as e:
            logger.error("Failed to update conversation context", error=str(e))

    def _analyze_sentiment(self, message_content: str) -> UserSentiment:
        """Analyze user sentiment from message content."""
        message_lower = message_content.lower()

        # Count sentiment indicators
        sentiment_scores = {}

        for sentiment, patterns in self.sentiment_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                sentiment_scores[sentiment] = score

        # Return sentiment with highest score, default to neutral
        if sentiment_scores:
            return max(sentiment_scores.items(), key=lambda x: x[1])[0]

        return UserSentiment.NEUTRAL

    def _extract_topics(self, message_content: str) -> List[str]:
        """Extract topics from message content."""
        # Simple keyword-based topic extraction
        # In production, this could use more sophisticated NLP

        topic_keywords = {
            "insurance": ["تأمين", "insurance", "وثيقة", "policy"],
            "payment": ["دفع", "payment", "فاتورة", "billing"],
            "login": ["تسجيل دخول", "login", "كلمة مرور", "password"],
            "renewal": ["تجديد", "renewal", "انتهاء", "expiry"],
            "claim": ["مطالبة", "claim", "حادث", "accident"],
            "registration": ["تسجيل", "registration", "حساب", "account"]
        }

        message_lower = message_content.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics

    def _extract_category_from_source(self, source: str) -> Optional[str]:
        """Extract category from source metadata."""
        # This would parse source metadata to extract category
        # For now, return None as placeholder
        return None

    def _update_interaction_patterns(
        self,
        conversation: ConversationContext,
        message: ConversationMessage
    ) -> None:
        """Update user interaction patterns."""
        patterns = conversation.interaction_patterns

        # Track message timing
        if "message_intervals" not in patterns:
            patterns["message_intervals"] = []

        if conversation.messages:
            last_message = conversation.messages[-2] if len(conversation.messages) > 1 else None
            if last_message:
                interval = (message.timestamp - last_message.timestamp).total_seconds()
                patterns["message_intervals"].append(interval)

                # Keep only recent intervals
                if len(patterns["message_intervals"]) > 10:
                    patterns["message_intervals"] = patterns["message_intervals"][-10:]

        # Track message length patterns
        if "message_lengths" not in patterns:
            patterns["message_lengths"] = []

        patterns["message_lengths"].append(len(message.content))
        if len(patterns["message_lengths"]) > 10:
            patterns["message_lengths"] = patterns["message_lengths"][-10:]

        # Track query complexity
        if message.message_type == MessageType.USER_QUERY:
            if "query_complexity" not in patterns:
                patterns["query_complexity"] = []

            complexity = self._calculate_query_complexity(message.content)
            patterns["query_complexity"].append(complexity)
            if len(patterns["query_complexity"]) > 10:
                patterns["query_complexity"] = patterns["query_complexity"][-10:]

    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        # Simple complexity calculation based on length and structure
        word_count = len(query.split())
        sentence_count = len([s for s in query.split('.') if s.strip()])
        question_marks = query.count('?') + query.count('؟')

        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count / 20) + (sentence_count / 5) + (question_marks / 3))
        return complexity

    async def _create_conversation_summary(self, conversation: ConversationContext) -> None:
        """Create conversation summary for context compression."""
        try:
            # Extract key information
            key_topics = list(set(conversation.topics))
            user_messages = [msg for msg in conversation.messages if msg.message_type == MessageType.USER_QUERY]
            ai_messages = [msg for msg in conversation.messages if msg.message_type == MessageType.AI_RESPONSE]

            # Create summary text
            summary_parts = []

            if key_topics:
                summary_parts.append(f"Topics discussed: {', '.join(key_topics)}")

            if user_messages:
                summary_parts.append(f"User asked {len(user_messages)} questions")

            if ai_messages:
                confidences = [msg.confidence_score for msg in ai_messages if msg.confidence_score]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    summary_parts.append(f"AI provided {len(ai_messages)} responses (avg confidence: {avg_confidence:.2f})")
                else:
                    summary_parts.append(f"AI provided {len(ai_messages)} responses")

            summary_parts.append(f"User sentiment: {conversation.user_sentiment.value}")

            # Determine resolution status
            resolution_status = "in_progress"
            if conversation.user_sentiment == UserSentiment.SATISFIED:
                resolution_status = "resolved"
            elif conversation.escalation_triggers:
                resolution_status = "escalated"

            # Create summary
            summary = ConversationSummary(
                conversation_id=conversation.conversation_id,
                summary_text="; ".join(summary_parts),
                key_topics=key_topics,
                resolution_status=resolution_status,
                user_satisfaction=conversation.user_sentiment.value
            )

            # Store summary
            self.conversation_summaries[conversation.conversation_id] = summary

            logger.info(
                "Conversation summary created",
                conversation_id=conversation.conversation_id,
                key_topics=key_topics,
                resolution_status=resolution_status
            )

        except Exception as e:
            logger.error("Failed to create conversation summary", error=str(e))

    def _calculate_session_duration(self, conversation: ConversationContext) -> int:
        """Calculate session duration in minutes."""
        duration = datetime.utcnow() - conversation.created_at
        return int(duration.total_seconds() / 60)

    def _generate_conversation_id(self, user_id: Optional[str], session_id: Optional[str]) -> str:
        """Generate unique conversation ID."""
        timestamp = datetime.utcnow().isoformat()
        base_string = f"{user_id or 'anonymous'}_{session_id or 'session'}_{timestamp}"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _generate_message_id(self, conversation_id: str) -> str:
        """Generate unique message ID."""
        timestamp = datetime.utcnow().isoformat()
        base_string = f"{conversation_id}_{timestamp}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]

    async def cleanup_idle_conversations(self) -> int:
        """Clean up idle conversations."""
        try:
            current_time = datetime.utcnow()
            idle_threshold = current_time - timedelta(minutes=self.idle_timeout_minutes)

            idle_conversations = []

            for conv_id, conversation in self.active_conversations.items():
                if conversation.updated_at < idle_threshold:
                    conversation.state = ConversationState.IDLE
                    idle_conversations.append(conv_id)

            # Remove idle conversations from active memory
            for conv_id in idle_conversations:
                del self.active_conversations[conv_id]

            logger.info(f"Cleaned up {len(idle_conversations)} idle conversations")
            return len(idle_conversations)

        except Exception as e:
            logger.error("Failed to cleanup idle conversations", error=str(e))
            return 0

    async def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics and metrics."""
        try:
            total_conversations = len(self.active_conversations)

            if total_conversations == 0:
                return {"total_conversations": 0}

            # Calculate metrics
            states = [conv.state.value for conv in self.active_conversations.values()]
            sentiments = [conv.user_sentiment.value for conv in self.active_conversations.values()]
            message_counts = [conv.total_messages for conv in self.active_conversations.values()]

            analytics = {
                "total_conversations": total_conversations,
                "conversation_states": {state: states.count(state) for state in set(states)},
                "user_sentiments": {sentiment: sentiments.count(sentiment) for sentiment in set(sentiments)},
                "avg_messages_per_conversation": sum(message_counts) / len(message_counts),
                "total_messages": sum(message_counts),
                "conversations_with_escalation_triggers": sum(
                    1 for conv in self.active_conversations.values() if conv.escalation_triggers
                )
            }

            return analytics

        except Exception as e:
            logger.error("Failed to get conversation analytics", error=str(e))
            return {"error": str(e)}


# Global instance
_conversation_memory_manager = None


def get_conversation_memory_manager() -> ConversationMemoryManager:
    """Get global conversation memory manager instance."""
    global _conversation_memory_manager
    if _conversation_memory_manager is None:
        _conversation_memory_manager = ConversationMemoryManager()
    return _conversation_memory_manager
