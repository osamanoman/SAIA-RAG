#!/usr/bin/env python3
"""
WhatsApp Formatting Test

This script demonstrates the new WhatsApp-specific formatting
features for better readability.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_whatsapp_formatting():
    """Test WhatsApp message formatting with different types of content."""
    print("🧪 Testing WhatsApp Message Formatting...")
    
    # Test cases with different content types
    test_cases = [
        {
            "name": "Services List",
            "query": "خدمات وازن",
            "expected_format": "bullet points and emojis"
        },
        {
            "name": "Troubleshooting Steps",
            "query": "كيف أحل مشكلة في النظام",
            "expected_format": "numbered steps converted to bullets"
        },
        {
            "name": "Setup Instructions",
            "query": "كيف أقوم بإعداد الحساب",
            "expected_format": "setup emoji and clean formatting"
        },
        {
            "name": "Billing Question",
            "query": "معلومات عن الفواتير",
            "expected_format": "billing emoji and structured response"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📱 Testing: {test_case['name']}")
        print(f"   Query: {test_case['query']}")
        print(f"   Expected: {test_case['expected_format']}")
        
        # Test payload
        test_payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "test_id",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "+966501234567",
                            "phone_number_id": "test_phone_id"
                        },
                        "messages": [{
                            "from": "966501234567",
                            "id": "test_message_id",
                            "timestamp": "1234567890",
                            "text": {
                                "body": test_case["query"]
                            },
                            "type": "text"
                        }]
                    },
                    "field": "messages"
                }]
            }]
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/whatsapp/simulate",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                rag_response = result.get('rag_response', {})
                
                if rag_response:
                    response_text = rag_response.get('response', '')
                    print(f"   ✅ Response received ({len(response_text)} chars)")
                    
                    # Check formatting features
                    has_bullets = '•' in response_text
                    has_emoji = any(emoji in response_text for emoji in ['🛠️', '🔧', '⚙️', '💰', '📋'])
                    has_clean_spacing = '\n\n' in response_text or '\n•' in response_text
                    
                    print(f"   📋 Bullet points: {'✅' if has_bullets else '❌'}")
                    print(f"   😊 Emojis: {'✅' if has_emoji else '❌'}")
                    print(f"   📏 Clean spacing: {'✅' if has_clean_spacing else '❌'}")
                    
                    # Show preview
                    preview = response_text[:150] + "..." if len(response_text) > 150 else response_text
                    print(f"   📝 Preview: {preview}")
                    
                else:
                    print("   ❌ No RAG response received")
                    
            else:
                print(f"   ❌ Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ WhatsApp Formatting Tests Completed!")
    print("\nExpected improvements:")
    print("• Clean bullet points (•) instead of numbers or asterisks")
    print("• Relevant emojis for different content types")
    print("• Better spacing and readability")
    print("• No unwanted escalation messages")

def main():
    """Run the formatting tests."""
    print("🚀 Starting WhatsApp Formatting Tests")
    print("=" * 60)
    
    test_whatsapp_formatting()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("If you see ✅ for bullet points, emojis, and clean spacing,")
    print("the WhatsApp formatting is working correctly!")

if __name__ == "__main__":
    main()
