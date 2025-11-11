#!/usr/bin/env python3
"""
Test script for conversational context management in SAIA-RAG.

Tests the addiction/harm case scenario with follow-up queries.
"""

import asyncio
import sys
from datetime import datetime

# Add app to path
sys.path.insert(0, '/Users/osama/Documents/apps/SAIA-RAG/SAIA-RAG')

from app.rag_service import get_rag_service
from app.conversation_memory import get_conversation_memory_manager


async def test_conversation_context():
    """Test conversation context retention with addiction/harm case."""
    
    print("=" * 80)
    print("SAIA-RAG Conversational Context Test")
    print("=" * 80)
    print()
    
    # Initialize services
    rag_service = get_rag_service()
    conversation_manager = get_conversation_memory_manager()
    
    # Generate unique conversation ID
    conversation_id = f"test-addiction-case-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    print(f"Conversation ID: {conversation_id}")
    print()
    
    # Test 1: Initial query with case description
    print("=" * 80)
    print("TEST 1: Initial Query (Case Description)")
    print("=" * 80)
    
    initial_query = """زوجة تطلب فسخ عقد الزواج بسبب إدمان الزوج وتعريضه الأسرة للخطر.

المدعية (ريم) متزوجة من المدعى عليه (ماجد) منذ عام 1440هـ، ولديهما طفلان.
خلال السنوات الأخيرة، أصبح الزوج مدمنًا على تعاطي المواد المخدّرة، مما أدى إلى تكرار اعتدائه على زوجته بالضرب والإهانة، وإهماله للنفقة والمنزل.

ريم تطلب:
1. فسخ عقد الزواج
2. النفقة الماضية والمستمرة
3. حضانة الأطفال مع تنظيم حق الزيارة"""
    
    print(f"Query: {initial_query[:100]}...")
    print()
    
    try:
        response1 = await rag_service.generate_response(
            query=initial_query,
            conversation_id=conversation_id
        )
        
        print(f"✅ Response generated")
        print(f"   Confidence: {response1['confidence']:.2f}")
        print(f"   Sources: {response1['sources_count']}")
        print(f"   Conversation aware: {response1.get('conversation_aware', False)}")
        print(f"   Response preview: {response1['response'][:200]}...")
        print()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Check conversation was stored
    conversation = await conversation_manager.get_conversation(conversation_id)
    if conversation:
        print(f"✅ Conversation stored: {len(conversation.messages)} messages")
        print()
    else:
        print(f"❌ Conversation not found")
        print()
    
    # Test 2: Follow-up query about alimony
    print("=" * 80)
    print("TEST 2: Follow-up Query (Alimony)")
    print("=" * 80)
    
    followup_query1 = "ماذا عن النفقة"
    
    print(f"Query: {followup_query1}")
    print()
    
    try:
        response2 = await rag_service.generate_response(
            query=followup_query1,
            conversation_id=conversation_id
        )
        
        print(f"✅ Response generated")
        print(f"   Confidence: {response2['confidence']:.2f}")
        print(f"   Sources: {response2['sources_count']}")
        print(f"   Conversation aware: {response2.get('conversation_aware', False)}")
        print(f"   Enhanced query: {response2.get('enhanced_query', 'N/A')[:150]}...")
        print(f"   Preprocessing: {response2.get('preprocessing_steps', [])}")
        print(f"   Response preview: {response2['response'][:200]}...")
        print()
        
        # Check if context reformulation was applied
        if 'context_reformulation' in response2.get('preprocessing_steps', []):
            print("✅ Context reformulation applied!")
        else:
            print("⚠️  Context reformulation NOT applied")
        print()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Test 3: Follow-up query about custody
    print("=" * 80)
    print("TEST 3: Follow-up Query (Custody)")
    print("=" * 80)
    
    followup_query2 = "ماذا عن الحضانة في هذه القضية"
    
    print(f"Query: {followup_query2}")
    print()
    
    try:
        response3 = await rag_service.generate_response(
            query=followup_query2,
            conversation_id=conversation_id
        )
        
        print(f"✅ Response generated")
        print(f"   Confidence: {response3['confidence']:.2f}")
        print(f"   Sources: {response3['sources_count']}")
        print(f"   Conversation aware: {response3.get('conversation_aware', False)}")
        print(f"   Enhanced query: {response3.get('enhanced_query', 'N/A')[:150]}...")
        print(f"   Preprocessing: {response3.get('preprocessing_steps', [])}")
        print(f"   Response preview: {response3['response'][:200]}...")
        print()
        
        # Check if context reformulation was applied
        if 'context_reformulation' in response3.get('preprocessing_steps', []):
            print("✅ Context reformulation applied!")
        else:
            print("⚠️  Context reformulation NOT applied")
        print()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Final conversation check
    conversation = await conversation_manager.get_conversation(conversation_id)
    if conversation:
        print("=" * 80)
        print("FINAL CONVERSATION STATE")
        print("=" * 80)
        print(f"Total messages: {len(conversation.messages)}")
        print(f"State: {conversation.state.value}")
        print(f"Topics: {conversation.topics}")
        print(f"Categories: {conversation.categories}")
        print()
        
        print("Message history:")
        for i, msg in enumerate(conversation.messages, 1):
            print(f"  {i}. [{msg.message_type.value}] {msg.content[:80]}...")
        print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_conversation_context())

