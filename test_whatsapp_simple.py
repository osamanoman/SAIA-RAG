#!/usr/bin/env python3
"""
Simple WhatsApp Integration Test

This script tests the basic WhatsApp functionality to ensure
the fixes for unwanted text are working.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_whatsapp_simulation():
    """Test WhatsApp message simulation with the fixes."""
    print("🧪 Testing WhatsApp Message Simulation...")
    
    # Test payload matching WhatsApp webhook format
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
                            "body": "خدمات وازن"
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
            print("✅ WhatsApp simulation successful")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            # Check RAG response
            rag_response = result.get('rag_response', {})
            if rag_response:
                response_text = rag_response.get('response', '')
                print(f"   Response length: {len(response_text)}")
                print(f"   Response preview: {response_text[:200]}...")
                
                # Check for unwanted text
                unwanted_phrases = [
                    "I understand your concern",
                    "أفهم قلقك",
                    "أريد التأكد من حصولك على أفضل مساعدة ممكنة",
                    "دعني أربطك بأخصائي",
                    "الخيارات:"
                ]
                
                found_unwanted = []
                for phrase in unwanted_phrases:
                    if phrase in response_text:
                        found_unwanted.append(phrase)
                
                if found_unwanted:
                    print("❌ Found unwanted text:")
                    for phrase in found_unwanted:
                        print(f"     - {phrase}")
                else:
                    print("✅ No unwanted text found - response is clean!")
                
                # Check confidence
                confidence = rag_response.get('confidence', 0)
                print(f"   Confidence: {confidence}")
                
            else:
                print("❌ No RAG response in simulation result")
                
        else:
            print(f"❌ WhatsApp simulation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
    
    return True

def test_whatsapp_status():
    """Test WhatsApp status endpoint."""
    print("\n🔧 Testing WhatsApp Status...")
    
    try:
        response = requests.get(f"{BASE_URL}/whatsapp/status", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ WhatsApp status endpoint accessible")
            print(f"   Configured: {result.get('configured', False)}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"❌ WhatsApp status failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Status test failed: {str(e)}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting WhatsApp Integration Tests")
    print("=" * 50)
    
    # Test 1: Status
    test_whatsapp_status()
    
    # Test 2: Simulation
    test_whatsapp_simulation()
    
    print("\n" + "=" * 50)
    print("✅ Tests completed!")
    print("\nIf you see 'No unwanted text found', the fixes are working!")
    print("If you see unwanted text, there may still be issues to resolve.")

if __name__ == "__main__":
    main()
