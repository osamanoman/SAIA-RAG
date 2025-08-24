"""
SAIA-RAG WhatsApp Business API Integration

WhatsApp Business API client for sending and receiving messages.
Integrates with the RAG system to provide AI-powered customer support via WhatsApp.
"""

import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from .config import get_settings, Settings

# Get logger
logger = structlog.get_logger()


class WhatsAppClient:
    """
    WhatsApp Business API client for SAIA-RAG integration.
    
    Provides functionality to:
    - Send text messages to WhatsApp users
    - Handle incoming webhook messages
    - Verify webhook authenticity
    - Format messages for WhatsApp display
    """
    
    def __init__(self, settings: Settings = None):
        """Initialize WhatsApp client with configuration."""
        self.settings = settings or get_settings()
        
        # WhatsApp API configuration
        self.access_token = self.settings.whatsapp_access_token
        self.phone_number_id = self.settings.whatsapp_phone_number_id
        self.verify_token = self.settings.whatsapp_verify_token
        
        # API endpoints
        self.base_url = "https://graph.facebook.com/v18.0"
        self.messages_url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        # Request headers
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(
            "WhatsApp client initialized",
            phone_number_id=self.phone_number_id,
            configured=self.is_configured()
        )
    
    def is_configured(self) -> bool:
        """Check if WhatsApp client is properly configured."""
        return all([
            self.access_token,
            self.phone_number_id,
            self.verify_token
        ])
    
    async def send_text_message(
        self,
        to: str,
        message: str,
        preview_url: bool = False
    ) -> Dict[str, Any]:
        """
        Send a text message to a WhatsApp user.
        
        Args:
            to: WhatsApp phone number (with country code, no + sign)
            message: Text message to send
            preview_url: Whether to show URL previews
            
        Returns:
            API response with message status
            
        Raises:
            Exception: If message sending fails
        """
        if not self.is_configured():
            raise Exception("WhatsApp client is not properly configured")
        
        try:
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {
                    "body": message,
                    "preview_url": preview_url
                }
            }
            
            response = requests.post(
                self.messages_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                "WhatsApp message sent successfully",
                to=to,
                message_id=result.get("messages", [{}])[0].get("id"),
                message_length=len(message)
            )
            
            return {
                "success": True,
                "message_id": result.get("messages", [{}])[0].get("id"),
                "status": "sent",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(
                "Failed to send WhatsApp message",
                to=to,
                error=str(e),
                status_code=getattr(e.response, 'status_code', None)
            )
            raise Exception(f"WhatsApp API error: {str(e)}")
        except Exception as e:
            logger.error("WhatsApp message sending failed", to=to, error=str(e))
            raise
    
    async def send_rag_response(
        self,
        to: str,
        rag_response: Dict[str, Any],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Send a RAG-generated response via WhatsApp with optional source citations.
        
        Args:
            to: WhatsApp phone number
            rag_response: RAG response dictionary from chat endpoint
            include_sources: Whether to include source citations
            
        Returns:
            Message sending result
        """
        try:
            # Format the main response
            message = rag_response["response"]
            
            # Add confidence indicator if low confidence
            confidence = rag_response.get("confidence", 0.0)
            if confidence < 0.5:
                message += f"\n\n_Note: I'm {confidence:.0%} confident about this answer. You may want to verify this information._"
            
            # Add source citations if requested and available
            if include_sources and rag_response.get("sources"):
                sources = rag_response["sources"][:3]  # Limit to 3 sources for WhatsApp
                if sources:
                    message += "\n\n📚 *Sources:*"
                    for i, source in enumerate(sources, 1):
                        title = source.get("title", "Document")
                        score = source.get("relevance_score", 0.0)
                        message += f"\n{i}. {title} (relevance: {score:.0%})"
            
            # Add processing info
            processing_time = rag_response.get("processing_time_ms", 0)
            if processing_time > 1000:
                message += f"\n\n_Response generated in {processing_time/1000:.1f}s_"
            
            # Send the formatted message
            return await self.send_text_message(to, message)
            
        except Exception as e:
            logger.error("Failed to send RAG response via WhatsApp", to=to, error=str(e))
            # Send a fallback message
            fallback_message = "I apologize, but I'm experiencing technical difficulties. Please try again later or contact support."
            return await self.send_text_message(to, fallback_message)
    
    def verify_webhook(self, mode: str, token: str, challenge: str) -> Optional[str]:
        """
        Verify WhatsApp webhook during setup.
        
        Args:
            mode: Verification mode from webhook
            token: Verification token from webhook
            challenge: Challenge string from webhook
            
        Returns:
            Challenge string if verification succeeds, None otherwise
        """
        if mode == "subscribe" and token == self.verify_token:
            logger.info("WhatsApp webhook verified successfully")
            return challenge
        else:
            logger.warning(
                "WhatsApp webhook verification failed",
                mode=mode,
                token_match=token == self.verify_token
            )
            return None
    
    def parse_webhook_message(self, webhook_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse incoming WhatsApp webhook message.
        
        Args:
            webhook_data: Raw webhook data from WhatsApp
            
        Returns:
            Parsed message data or None if not a text message
        """
        try:
            entry = webhook_data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            
            # Check if this is a message event
            if "messages" not in value:
                return None
            
            message = value["messages"][0]
            contact = value.get("contacts", [{}])[0]
            
            # Only handle text messages for now
            if message.get("type") != "text":
                logger.info("Received non-text message", message_type=message.get("type"))
                return None
            
            parsed_message = {
                "message_id": message.get("id"),
                "from": message.get("from"),
                "timestamp": message.get("timestamp"),
                "text": message.get("text", {}).get("body"),
                "contact_name": contact.get("profile", {}).get("name"),
                "phone_number": message.get("from")
            }
            
            logger.info(
                "WhatsApp message parsed",
                from_number=parsed_message["from"],
                message_length=len(parsed_message["text"]) if parsed_message["text"] else 0
            )
            
            return parsed_message
            
        except Exception as e:
            logger.error("Failed to parse WhatsApp webhook message", error=str(e))
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check WhatsApp API connectivity and configuration.
        
        Returns:
            Health status dictionary
        """
        try:
            if not self.is_configured():
                return {
                    "status": "unhealthy",
                    "error": "WhatsApp client not configured",
                    "configured": False
                }
            
            # Test API connectivity with a simple request
            test_url = f"{self.base_url}/{self.phone_number_id}"
            response = requests.get(
                test_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "configured": True,
                    "phone_number_id": self.phone_number_id,
                    "api_version": "v18.0"
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"API returned status {response.status_code}",
                    "configured": True
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "configured": self.is_configured()
            }


def get_whatsapp_client() -> WhatsAppClient:
    """
    Get WhatsApp client instance.
    
    Returns:
        WhatsApp client instance
    """
    return WhatsAppClient()


# Export for easy importing
__all__ = ["WhatsAppClient", "get_whatsapp_client"]
