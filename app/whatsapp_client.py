"""
SAIA-RAG WhatsApp Business API Integration

WhatsApp Business API client for sending and receiving messages.
Integrates with the RAG system to provide AI-powered customer support via WhatsApp.
"""

import json
import requests
import re
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
            configured=self.is_configured(),
            has_access_token=bool(self.access_token),
            has_phone_number_id=bool(self.phone_number_id),
            has_verify_token=bool(self.verify_token)
        )
    
    def is_configured(self) -> bool:
        """Check if WhatsApp client is properly configured."""
        configured = all([
            self.access_token,
            self.phone_number_id,
            self.verify_token
        ])
        
        if not configured:
            logger.warning("WhatsApp client not fully configured",
                          has_access_token=bool(self.access_token),
                          has_phone_number_id=bool(self.phone_number_id),
                          has_verify_token=bool(self.verify_token))
        
        return configured
    
    def _format_phone_number(self, phone: str) -> str:
        """
        Format phone number for WhatsApp API.
        
        WhatsApp API expects phone numbers without '+' and with country code.
        Example: +966501234567 -> 966501234567
        
        Args:
            phone: Phone number (with or without +)
            
        Returns:
            Formatted phone number for WhatsApp API
        """
        # Remove all non-digit characters
        cleaned = re.sub(r'[^\d]', '', phone)
        
        # Ensure it starts with country code (at least 7 digits)
        if len(cleaned) < 7:
            raise ValueError(f"Invalid phone number: {phone}")
        
        # If it starts with 00, remove it (international format)
        if cleaned.startswith('00'):
            cleaned = cleaned[2:]
        
        # If it starts with +, it was already removed by regex
        
        logger.debug("Phone number formatted", original=phone, formatted=cleaned)
        return cleaned
    
    def _validate_message(self, message: str) -> str:
        """
        Validate and clean message for WhatsApp.
        
        Args:
            message: Raw message text
            
        Returns:
            Cleaned and validated message
            
        Raises:
            ValueError: If message is invalid
        """
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")
        
        # WhatsApp has a 4096 character limit for text messages
        if len(message) > 4096:
            # Truncate and add ellipsis
            message = message[:4093] + "..."
            logger.warning("Message truncated due to length limit", 
                          original_length=len(message), 
                          truncated_length=4096)
        
        # Remove any control characters that might cause issues
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', message)
        
        return cleaned.strip()
    
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
            # Format phone number
            formatted_phone = self._format_phone_number(to)
            
            # Validate and clean message
            cleaned_message = self._validate_message(message)
            
            payload = {
                "messaging_product": "whatsapp",
                "to": formatted_phone,
                "type": "text",
                "text": {
                    "body": cleaned_message,
                    "preview_url": preview_url
                }
            }
            
            logger.info("Sending WhatsApp message", 
                       to=formatted_phone, 
                       message_length=len(cleaned_message))
            
            response = requests.post(
                self.messages_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Handle different response status codes
            if response.status_code == 200:
                result = response.json()
                message_id = result.get("messages", [{}])[0].get("id")
                
                logger.info(
                    "WhatsApp message sent successfully",
                    to=formatted_phone,
                    message_id=message_id,
                    message_length=len(cleaned_message)
                )
                
                return {
                    "success": True,
                    "message_id": message_id,
                    "status": "sent",
                    "timestamp": datetime.utcnow().isoformat()
                }
            elif response.status_code == 400:
                # Bad request - usually invalid phone number or message
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Bad request")
                raise Exception(f"WhatsApp API error: {error_msg}")
            elif response.status_code == 401:
                # Unauthorized - invalid access token
                raise Exception("WhatsApp API error: Invalid access token")
            elif response.status_code == 403:
                # Forbidden - phone number not in allowed list
                raise Exception("WhatsApp API error: Phone number not allowed")
            elif response.status_code == 429:
                # Rate limited
                raise Exception("WhatsApp API error: Rate limited - try again later")
            else:
                # Other errors
                error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                raise Exception(f"WhatsApp API error: HTTP {response.status_code} - {error_data}")
            
        except requests.exceptions.RequestException as e:
            logger.error(
                "Failed to send WhatsApp message",
                to=to,
                error=str(e),
                status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
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
            # Format the main response (clean, no extra messages)
            message = rag_response["response"]

            # Send the clean message without confidence warnings or processing time
            return await self.send_text_message(to, message)
            
        except Exception as e:
            logger.error("Failed to send RAG response via WhatsApp", to=to, error=str(e))
            # Re-raise the exception instead of sending fallback message
            raise Exception(f"WhatsApp message sending failed: {str(e)}")
    
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
            logger.info("Parsing WhatsApp webhook message",
                        webhook_keys=list(webhook_data.keys()),
                        webhook_data_type=type(webhook_data).__name__)

            # Validate webhook structure
            if not isinstance(webhook_data, dict):
                logger.warning("Webhook data is not a dictionary", data_type=type(webhook_data).__name__)
                return None

            # Step 1: Extract entry
            entries = webhook_data.get("entry", [])
            if not entries:
                logger.info("No entries in webhook data")
                return None

            entry = entries[0]
            if not isinstance(entry, dict):
                logger.warning("Entry is not a dictionary", entry_type=type(entry).__name__)
                return None

            # Step 2: Extract changes
            changes = entry.get("changes", [])
            if not changes:
                logger.info("No changes in entry")
                return None

            change = changes[0]
            if not isinstance(change, dict):
                logger.warning("Change is not a dictionary", change_type=type(change).__name__)
                return None

            value = change.get("value", {})
            if not isinstance(value, dict):
                logger.warning("Value is not a dictionary", value_type=type(value).__name__)
                return None

            # Step 3: Check for messages
            if "messages" not in value:
                logger.info("No messages in value", value_keys=list(value.keys()))
                return None

            messages = value["messages"]
            if not messages or not isinstance(messages, list):
                logger.info("Messages is empty or not a list", messages_type=type(messages).__name__)
                return None

            message = messages[0]
            if not isinstance(message, dict):
                logger.warning("Message is not a dictionary", message_type=type(message).__name__)
                return None

            # Step 4: Check message type
            message_type = message.get("type")
            if message_type != "text":
                logger.info("Message is not text type", message_type=message_type)
                return None

            # Step 5: Extract message data
            text_data = message.get("text", {})
            if not isinstance(text_data, dict):
                logger.warning("Text data is not a dictionary", text_data_type=type(text_data).__name__)
                return None

            message_body = text_data.get("body")
            if not message_body:
                logger.info("No body in text data")
                return None

            # Validate required fields
            required_fields = ["id", "from", "timestamp"]
            for field in required_fields:
                if field not in message:
                    logger.warning(f"Missing required field: {field}")
                    return None

            parsed_message = {
                "message_id": message.get("id"),
                "from": message.get("from"),
                "timestamp": message.get("timestamp"),
                "text": message_body,
                "contact_name": None,  # Simplified for now
                "phone_number": message.get("from")
            }

            logger.info("WhatsApp message parsed successfully",
                        message_id=parsed_message["message_id"],
                        from_number=parsed_message["from"],
                        text_length=len(parsed_message["text"]))

            return parsed_message

        except Exception as e:
            logger.error("Failed to parse WhatsApp webhook message",
                        error=str(e),
                        error_type=type(e).__name__,
                        webhook_data_keys=list(webhook_data.keys()) if webhook_data else None)
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
                    "configured": False,
                    "missing_fields": []
                }
            
            # Check which fields are missing
            missing_fields = []
            if not self.access_token:
                missing_fields.append("access_token")
            if not self.phone_number_id:
                missing_fields.append("phone_number_id")
            if not self.verify_token:
                missing_fields.append("verify_token")
            
            if missing_fields:
                return {
                    "status": "unhealthy",
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                    "configured": False,
                    "missing_fields": missing_fields
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
                    "api_version": "v18.0",
                    "missing_fields": []
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"API returned status {response.status_code}",
                    "configured": True,
                    "missing_fields": []
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "configured": self.is_configured(),
                "missing_fields": []
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
