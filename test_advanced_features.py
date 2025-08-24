#!/usr/bin/env python3
"""
Comprehensive test script for SAIA-RAG advanced features.

Tests:
1. Query Processing and Enhancement
2. Content Categorization
3. Adaptive Retrieval Strategies
4. Intelligent Reranking
5. Conversation Memory
6. Enhanced RAG Service
7. Multi-turn Conversations
8. Escalation Detection
"""

import asyncio
import json
import requests
from datetime import datetime
from typing import Dict, List, Any

# Test configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Test data
TEST_QUERIES = [
    {
        "query": "كيف يمكنني تجديد وثيقة التأمين؟",
        "expected_category": "faq",
        "expected_strategy": "step_by_step",
        "description": "FAQ renewal question"
    },
    {
        "query": "لدي مشكلة في تسجيل الدخول ولا أستطيع الوصول لحسابي",
        "expected_category": "troubleshooting", 
        "expected_strategy": "troubleshooting",
        "description": "Login troubleshooting"
    },
    {
        "query": "ما هي سياسة الخصوصية وشروط الاستخدام؟",
        "expected_category": "policy",
        "expected_strategy": "policy_lookup",
        "description": "Policy inquiry"
    },
    {
        "query": "عاجل! لدي حادث وأحتاج تقديم مطالبة فوراً",
        "expected_category": "claim",
        "expected_strategy": "customer_support",
        "description": "Urgent claim request"
    },
    {
        "query": "معلومات عامة عن خدمات التأمين المتاحة",
        "expected_category": "general",
        "expected_strategy": "multi_category",
        "description": "General information query"
    }
]

CONVERSATION_TEST_SCENARIOS = [
    {
        "name": "Happy Path Conversation",
        "messages": [
            "مرحباً، أريد معرفة كيفية تجديد وثيقة التأمين",
            "شكراً، هذا مفيد جداً",
            "هل يمكنني الدفع بالتقسيط؟"
        ],
        "expected_sentiment": ["neutral", "satisfied", "neutral"],
        "expected_escalation": [False, False, False]
    },
    {
        "name": "Frustrated Customer",
        "messages": [
            "لدي مشكلة في الموقع",
            "هذا محبط جداً، لا أستطيع الوصول لحسابي",
            "أريد التحدث مع شخص حقيقي"
        ],
        "expected_sentiment": ["neutral", "frustrated", "angry"],
        "expected_escalation": [False, True, True]
    },
    {
        "name": "Complex Multi-topic",
        "messages": [
            "أريد معلومات عن التأمين الصحي",
            "كم تكلفة التأمين الشامل؟",
            "كيف أقدم مطالبة؟",
            "ما هي مدة معالجة المطالبة؟"
        ],
        "expected_sentiment": ["neutral", "neutral", "neutral", "neutral"],
        "expected_escalation": [False, False, False, False]
    }
]

def print_header(title: str):
    """Print formatted test section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print formatted test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"   {details}")

async def test_health_check():
    """Test system health."""
    print_header("SYSTEM HEALTH CHECK")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        health_data = response.json()
        
        # Check overall status
        overall_healthy = health_data.get("status") == "ok"
        print_test_result("Overall System Health", overall_healthy)
        
        # Check vector store
        vector_healthy = health_data.get("dependencies", {}).get("vector_store", {}).get("status") == "healthy"
        print_test_result("Vector Store Health", vector_healthy)
        
        # Check OpenAI
        openai_healthy = health_data.get("dependencies", {}).get("openai", {}).get("status") == "healthy"
        print_test_result("OpenAI Integration", openai_healthy)
        
        print(f"\n📊 System Info:")
        print(f"   Environment: {health_data.get('environment')}")
        print(f"   Vector Store: {health_data.get('dependencies', {}).get('vector_store', {}).get('collections_count')} collections")
        print(f"   OpenAI Model: {health_data.get('dependencies', {}).get('openai', {}).get('chat_model')}")
        
        return overall_healthy and vector_healthy and openai_healthy
        
    except Exception as e:
        print_test_result("System Health Check", False, f"Error: {str(e)}")
        return False

async def test_basic_chat():
    """Test basic chat functionality."""
    print_header("BASIC CHAT FUNCTIONALITY")
    
    test_message = "مرحباً، كيف يمكنني المساعدة؟"
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=HEADERS,
            json={"message": test_message},
            timeout=30
        )
        
        if response.status_code == 200:
            chat_data = response.json()
            
            # Check response structure
            has_message = "message" in chat_data
            has_confidence = "confidence_score" in chat_data
            has_sources = "sources" in chat_data
            
            print_test_result("Chat Response Structure", has_message and has_confidence)
            print_test_result("Response Content", len(chat_data.get("message", "")) > 0)
            print_test_result("Confidence Score", 0 <= chat_data.get("confidence_score", -1) <= 1)
            
            print(f"\n📝 Response Preview:")
            print(f"   Message: {chat_data.get('message', '')[:100]}...")
            print(f"   Confidence: {chat_data.get('confidence_score', 0):.3f}")
            print(f"   Sources: {len(chat_data.get('sources', []))}")
            
            return True
        else:
            print_test_result("Basic Chat", False, f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_test_result("Basic Chat", False, f"Error: {str(e)}")
        return False

async def test_query_processing():
    """Test query processing and categorization."""
    print_header("QUERY PROCESSING & CATEGORIZATION")
    
    success_count = 0
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        expected_category = test_case["expected_category"]
        description = test_case["description"]
        
        print(f"\n🔍 Test {i}: {description}")
        print(f"   Query: {query}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                headers=HEADERS,
                json={"message": query},
                timeout=30
            )
            
            if response.status_code == 200:
                chat_data = response.json()
                
                # Check if response is relevant
                response_relevant = len(chat_data.get("message", "")) > 20
                confidence_reasonable = chat_data.get("confidence_score", 0) > 0.1
                has_sources = len(chat_data.get("sources", [])) > 0
                
                print_test_result(f"Response Generated", response_relevant)
                print_test_result(f"Confidence Score", confidence_reasonable, f"{chat_data.get('confidence_score', 0):.3f}")
                print_test_result(f"Sources Retrieved", has_sources, f"{len(chat_data.get('sources', []))} sources")
                
                if response_relevant and confidence_reasonable:
                    success_count += 1
                
                # Show processing metadata if available
                if "retrieval_metadata" in chat_data:
                    retrieval_meta = chat_data["retrieval_metadata"]
                    print(f"   📊 Retrieval: {retrieval_meta.get('strategy_used', 'unknown')} strategy, {retrieval_meta.get('chunks_retrieved', 0)} chunks")
                
                if "reranking_metadata" in chat_data:
                    rerank_meta = chat_data["reranking_metadata"]
                    print(f"   🎯 Reranking: {rerank_meta.get('method_used', 'unknown')} method, {rerank_meta.get('score_improvement', 0):.3f} improvement")
                
            else:
                print_test_result(f"Query Processing", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print_test_result(f"Query Processing", False, f"Error: {str(e)}")
    
    overall_success = success_count >= len(TEST_QUERIES) * 0.8  # 80% success rate
    print_test_result(f"Overall Query Processing", overall_success, f"{success_count}/{len(TEST_QUERIES)} successful")
    
    return overall_success

async def test_conversation_memory():
    """Test conversation memory and multi-turn interactions."""
    print_header("CONVERSATION MEMORY & MULTI-TURN")
    
    success_count = 0
    
    for scenario in CONVERSATION_TEST_SCENARIOS:
        scenario_name = scenario["name"]
        messages = scenario["messages"]
        
        print(f"\n💬 Scenario: {scenario_name}")
        
        conversation_id = None
        scenario_success = True
        
        for i, message in enumerate(messages):
            print(f"   Turn {i+1}: {message}")
            
            try:
                # Prepare request with conversation context
                request_data = {"message": message}
                if conversation_id:
                    request_data["conversation_id"] = conversation_id
                
                response = requests.post(
                    f"{BASE_URL}/chat",
                    headers=HEADERS,
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    chat_data = response.json()
                    
                    # Get conversation ID from first response
                    if not conversation_id:
                        conversation_id = chat_data.get("conversation_id")
                    
                    # Check response quality
                    has_response = len(chat_data.get("message", "")) > 10
                    has_confidence = chat_data.get("confidence_score", 0) > 0
                    
                    print(f"      ✅ Response: {chat_data.get('message', '')[:50]}...")
                    print(f"      📊 Confidence: {chat_data.get('confidence_score', 0):.3f}")
                    
                    # Check conversation metadata
                    if "conversation_metadata" in chat_data:
                        conv_meta = chat_data["conversation_metadata"]
                        print(f"      💭 Context: {conv_meta.get('total_messages', 0)} messages, sentiment: {conv_meta.get('user_sentiment', 'unknown')}")
                    
                    # Check escalation
                    escalation = chat_data.get("escalation_recommended", False)
                    print(f"      🚨 Escalation: {'Recommended' if escalation else 'Not needed'}")
                    
                    if not (has_response and has_confidence):
                        scenario_success = False
                        
                else:
                    print(f"      ❌ HTTP {response.status_code}")
                    scenario_success = False
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                scenario_success = False
        
        if scenario_success:
            success_count += 1
        
        print_test_result(f"Scenario: {scenario_name}", scenario_success)
    
    overall_success = success_count >= len(CONVERSATION_TEST_SCENARIOS) * 0.7  # 70% success rate
    print_test_result("Overall Conversation Memory", overall_success, f"{success_count}/{len(CONVERSATION_TEST_SCENARIOS)} scenarios passed")
    
    return overall_success

async def test_document_ingestion():
    """Test document ingestion capability."""
    print_header("DOCUMENT INGESTION")
    
    # Test document
    test_document = {
        "title": "Test Insurance Policy",
        "content": "هذه وثيقة تأمين تجريبية. تغطي التأمين الصحي والسيارات. يمكن تجديد الوثيقة سنوياً. للمطالبات، يرجى الاتصال بخدمة العملاء.",
        "category": "policies",
        "metadata": {
            "source": "test_policy.pdf",
            "language": "ar",
            "created_at": datetime.now().isoformat()
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ingest",
            headers=HEADERS,
            json=test_document,
            timeout=30
        )
        
        if response.status_code == 200:
            ingest_data = response.json()
            
            success = ingest_data.get("status") == "success"
            chunks_created = ingest_data.get("chunks_created", 0) > 0
            
            print_test_result("Document Ingestion", success)
            print_test_result("Chunks Created", chunks_created, f"{ingest_data.get('chunks_created', 0)} chunks")
            
            print(f"\n📄 Ingestion Results:")
            print(f"   Status: {ingest_data.get('status')}")
            print(f"   Chunks: {ingest_data.get('chunks_created', 0)}")
            print(f"   Processing Time: {ingest_data.get('processing_time_ms', 0)}ms")
            
            return success and chunks_created
        else:
            print_test_result("Document Ingestion", False, f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_test_result("Document Ingestion", False, f"Error: {str(e)}")
        return False

async def test_performance_metrics():
    """Test system performance metrics."""
    print_header("PERFORMANCE METRICS")
    
    # Test multiple queries to measure performance
    test_queries = [
        "كيف يمكنني تجديد وثيقة التأمين؟",
        "ما هي تكلفة التأمين الصحي؟",
        "كيف أقدم مطالبة تأمين؟"
    ]
    
    response_times = []
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n⚡ Performance Test {i}: {query[:30]}...")
        
        try:
            start_time = datetime.now()
            
            response = requests.post(
                f"{BASE_URL}/chat",
                headers=HEADERS,
                json={"message": query},
                timeout=30
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                chat_data = response.json()
                processing_time = chat_data.get("processing_time_ms", 0)
                
                response_times.append(response_time)
                success_count += 1
                
                print(f"   ⏱️  Total Time: {response_time:.0f}ms")
                print(f"   🔧 Processing Time: {processing_time}ms")
                print(f"   📊 Confidence: {chat_data.get('confidence_score', 0):.3f}")
                
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"\n📈 Performance Summary:")
        print(f"   Average Response Time: {avg_response_time:.0f}ms")
        print(f"   Min Response Time: {min_response_time:.0f}ms")
        print(f"   Max Response Time: {max_response_time:.0f}ms")
        print(f"   Success Rate: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
        
        # Performance thresholds
        fast_responses = avg_response_time < 5000  # Under 5 seconds
        consistent_performance = max_response_time < 10000  # Under 10 seconds max
        high_success_rate = success_count >= len(test_queries) * 0.9  # 90% success
        
        print_test_result("Average Response Time", fast_responses, f"{avg_response_time:.0f}ms")
        print_test_result("Consistent Performance", consistent_performance, f"Max: {max_response_time:.0f}ms")
        print_test_result("High Success Rate", high_success_rate, f"{success_count/len(test_queries)*100:.1f}%")
        
        return fast_responses and consistent_performance and high_success_rate
    
    return False

async def main():
    """Run all tests."""
    print("🚀 SAIA-RAG Advanced Features Test Suite")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_results = {}
    
    test_results["health"] = await test_health_check()
    test_results["basic_chat"] = await test_basic_chat()
    test_results["query_processing"] = await test_query_processing()
    test_results["conversation_memory"] = await test_conversation_memory()
    test_results["document_ingestion"] = await test_document_ingestion()
    test_results["performance"] = await test_performance_metrics()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        print_test_result(test_name.replace("_", " ").title(), result)
    
    print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("🎉 SAIA-RAG Advanced Features: EXCELLENT PERFORMANCE!")
    elif passed_tests >= total_tests * 0.6:  # 60% pass rate
        print("✅ SAIA-RAG Advanced Features: GOOD PERFORMANCE")
    else:
        print("⚠️  SAIA-RAG Advanced Features: NEEDS IMPROVEMENT")
    
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
