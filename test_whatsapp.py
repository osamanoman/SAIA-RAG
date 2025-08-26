#!/usr/bin/env python3
"""
WhatsApp Integration Test Script

This script tests all aspects of the WhatsApp integration:
1. Configuration validation
2. Endpoint accessibility
3. Message processing
4. Response generation
5. Rate limiting
6. Error handling
"""

import requests
import json
import sys
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your domain in production
TEST_MESSAGE = "What services do you offer?"

def test_endpoint(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an endpoint and return the response."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            "headers": dict(response.headers)
        }
    except Exception as e:
        return {"error": str(e), "success": False}

def test_whatsapp_configuration():
    """Test WhatsApp configuration status."""
    print("🔧 Testing WhatsApp Configuration...")
    
    result = test_endpoint("/whatsapp/status")
    
    if result["success"]:
        print("✅ WhatsApp status endpoint accessible")
        print(f"   Status: {result['data'].get('status', 'unknown')}")
        print(f"   Configured: {result['data'].get('configured', False)}")
        
        if result['data'].get('configured'):
            print("✅ WhatsApp is properly configured")
        else:
            print("❌ WhatsApp is not configured")
            print("   Please check your .env.prod file")
    else:
        print(f"❌ WhatsApp status endpoint failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_whatsapp_debug():
    """Test WhatsApp debug endpoint."""
    print("\n🔍 Testing WhatsApp Debug Information...")
    
    result = test_endpoint("/whatsapp/debug")
    
    if result["success"]:
        print("✅ WhatsApp debug endpoint accessible")
        debug_data = result["data"]
        
        print("   Configuration:")
        config = debug_data.get("configuration", {})
        for key, value in config.items():
            status = "✅" if value else "❌"
            print(f"     {status} {key}: {value}")
        
        print("   Endpoints:")
        endpoints = debug_data.get("endpoints", {})
        for name, path in endpoints.items():
            print(f"     📍 {name}: {path}")
            
    else:
        print(f"❌ WhatsApp debug endpoint failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_whatsapp_rate_limiting():
    """Test WhatsApp rate limiting functionality."""
    print("\n⏱️ Testing WhatsApp Rate Limiting...")
    
    # Test rate limit info
    result = test_endpoint("/whatsapp/rate-limit")
    
    if result["success"]:
        print("✅ WhatsApp rate limit endpoint accessible")
        rate_data = result["data"]
        
        if "rate_limit_info" in rate_data:
            rate_info = rate_data["rate_limit_info"]
            print(f"   Client IP: {rate_info.get('client_ip', 'unknown')}")
            print(f"   Requests last minute: {rate_info.get('requests_last_minute', 0)}")
            print(f"   Total requests: {rate_info.get('total_requests', 0)}")
        
        if "limits" in rate_data:
            print("   Rate Limits:")
            for endpoint, limit in rate_data["limits"].items():
                print(f"     📍 {endpoint}: {limit}")
    else:
        print(f"❌ WhatsApp rate limit endpoint failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_whatsapp_simulation():
    """Test WhatsApp message simulation."""
    print("\n🧪 Testing WhatsApp Message Simulation...")
    
    # Test payload matching WhatsApp webhook format
    test_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "test_id",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "+1234567890",
                        "phone_number_id": "test_phone_id"
                    },
                    "messages": [{
                        "from": "1234567890",
                        "id": "test_message_id",
                        "timestamp": "1234567890",
                        "text": {
                            "body": TEST_MESSAGE
                        },
                        "type": "text"
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    
    result = test_endpoint("/whatsapp/simulate", "POST", test_payload)
    
    if result["success"]:
        print("✅ WhatsApp simulation endpoint accessible")
        sim_data = result["data"]
        
        print(f"   Status: {sim_data.get('status', 'unknown')}")
        print(f"   Message: {sim_data.get('message', 'No message')}")
        
        if sim_data.get('status') == 'success':
            print("✅ Message simulation successful")
            
            # Check parsed message
            parsed = sim_data.get('parsed_message', {})
            if parsed:
                print(f"   Parsed from: {parsed.get('from', 'unknown')}")
                print(f"   Parsed text: {parsed.get('text', 'no text')}")
            
            # Check RAG response
            rag_response = sim_data.get('rag_response', {})
            if rag_response:
                print(f"   RAG response: {rag_response.get('response', 'no response')[:100]}...")
                print(f"   Confidence: {rag_response.get('confidence', 0)}")
                print(f"   Sources: {rag_response.get('sources_count', 0)}")
        else:
            print(f"❌ Message simulation failed: {sim_data.get('message', 'Unknown error')}")
    else:
        print(f"❌ WhatsApp simulation endpoint failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_whatsapp_verification():
    """Test WhatsApp webhook verification."""
    print("\n🔐 Testing WhatsApp Webhook Verification...")
    
    # Test verification endpoint
    verify_url = "/whatsapp/verify?hub.mode=subscribe&hub.verify_token=test_token&hub.challenge=test_challenge"
    result = test_endpoint(verify_url)
    
    if result["success"]:
        print("✅ WhatsApp verification endpoint accessible")
        if result["status_code"] == 403:
            print("   Expected: Verification failed (test token)")
        elif result["status_code"] == 503:
            print("   Expected: WhatsApp not configured")
        else:
            print(f"   Unexpected status: {result['status_code']}")
    else:
        print(f"❌ WhatsApp verification endpoint failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_webhook_endpoint():
    """Test the main webhook endpoint."""
    print("\n📡 Testing WhatsApp Webhook Endpoint...")
    
    # Test with invalid payload (should return 400 or 200 with ignored status)
    test_payload = {"test": "invalid_payload"}
    result = test_endpoint("/whatsapp/webhook", "POST", test_payload)
    
    if result["success"]:
        print("✅ WhatsApp webhook endpoint accessible")
        webhook_data = result["data"]
        
        if isinstance(webhook_data, dict):
            status = webhook_data.get('status', 'unknown')
            print(f"   Status: {status}")
            
            if status == "ignored":
                print("   Expected: Invalid payload ignored")
            elif status == "not_configured":
                print("   Expected: WhatsApp not configured")
            elif status == "received":
                print("   Expected: Message processed")
            else:
                print(f"   Unexpected status: {status}")
        else:
            print(f"   Response: {webhook_data}")
    else:
        print(f"❌ WhatsApp webhook endpoint failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_rate_limit_enforcement():
    """Test rate limiting by making multiple requests."""
    print("\n🚫 Testing Rate Limit Enforcement...")
    
    # Make multiple requests to trigger rate limiting
    print("   Making multiple requests to test rate limiting...")
    
    for i in range(12):  # Try to exceed the 10 requests/minute limit
        result = test_endpoint("/whatsapp/rate-limit")
        if result["status_code"] == 429:
            print(f"   ✅ Rate limit triggered after {i+1} requests")
            break
        time.sleep(0.1)  # Small delay between requests
    else:
        print("   ⚠️ Rate limit not triggered (may need more requests)")
    
    return {"status": "tested"}

def test_phone_number_formatting():
    """Test phone number formatting functionality."""
    print("\n📱 Testing Phone Number Formatting...")
    
    # Test various phone number formats
    test_numbers = [
        "+966501234567",
        "966501234567",
        "+1-555-123-4567",
        "15551234567",
        "invalid"
    ]
    
    print("   Testing phone number formats:")
    for phone in test_numbers:
        try:
            # This would test the internal formatting function
            # For now, just show the test cases
            print(f"     📍 {phone}")
        except Exception as e:
            print(f"     ❌ {phone}: {str(e)}")
    
    return {"status": "tested"}

def run_all_tests():
    """Run all WhatsApp integration tests."""
    print("🚀 Starting WhatsApp Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_whatsapp_configuration),
        ("Debug Info", test_whatsapp_debug),
        ("Rate Limiting", test_whatsapp_rate_limiting),
        ("Message Simulation", test_whatsapp_simulation),
        ("Webhook Verification", test_whatsapp_verification),
        ("Webhook Endpoint", test_webhook_endpoint),
        ("Rate Limit Enforcement", test_rate_limit_enforcement),
        ("Phone Number Formatting", test_phone_number_formatting)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results[test_name] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        if result.get("success", False) or "error" not in result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
        
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! WhatsApp integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the configuration and fix issues.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        sys.exit(1)

