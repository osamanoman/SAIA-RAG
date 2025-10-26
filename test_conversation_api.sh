#!/bin/bash
# Test conversational context via API

echo "================================================================================"
echo "SAIA-RAG Conversational Context API Test"
echo "================================================================================"
echo ""

# API Key from .env.prod
API_KEY="saia-rag-production-api-key-2024-secure-random-key-12345"

# Generate unique conversation ID
CONVERSATION_ID="test-addiction-$(date +%Y%m%d%H%M%S)"
echo "Conversation ID: $CONVERSATION_ID"
echo ""

# Test 1: Initial query with case description
echo "================================================================================"
echo "TEST 1: Initial Query (Case Description)"
echo "================================================================================"
echo ""

INITIAL_QUERY='زوجة تطلب فسخ عقد الزواج بسبب إدمان الزوج وتعريضه الأسرة للخطر. المدعية (ريم) متزوجة من المدعى عليه (ماجد) منذ عام 1440هـ، ولديهما طفلان. خلال السنوات الأخيرة، أصبح الزوج مدمنًا على تعاطي المواد المخدّرة، مما أدى إلى تكرار اعتدائه على زوجته بالضرب والإهانة، وإهماله للنفقة والمنزل. ريم تطلب: 1. فسخ عقد الزواج 2. النفقة الماضية والمستمرة 3. حضانة الأطفال مع تنظيم حق الزيارة'

echo "Sending initial query..."
RESPONSE1=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"message\": \"$INITIAL_QUERY\",
    \"conversation_id\": \"$CONVERSATION_ID\"
  }")

echo "Response:"
echo "$RESPONSE1" | python3 -m json.tool | head -50
echo ""

# Extract confidence and sources count
CONFIDENCE1=$(echo "$RESPONSE1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence', 0))")
SOURCES1=$(echo "$RESPONSE1" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('sources', [])))")

echo "✅ Initial query processed"
echo "   Confidence: $CONFIDENCE1"
echo "   Sources: $SOURCES1"
echo ""

sleep 2

# Test 2: Follow-up query about alimony
echo "================================================================================"
echo "TEST 2: Follow-up Query (Alimony)"
echo "================================================================================"
echo ""

FOLLOWUP1='ماذا عن النفقة'

echo "Sending follow-up query: $FOLLOWUP1"
RESPONSE2=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"message\": \"$FOLLOWUP1\",
    \"conversation_id\": \"$CONVERSATION_ID\"
  }")

echo "Response:"
echo "$RESPONSE2" | python3 -m json.tool | head -50
echo ""

# Check for context reformulation
PREPROCESSING=$(echo "$RESPONSE2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('preprocessing_steps', []))" 2>/dev/null)

echo "✅ Follow-up query processed"
if echo "$PREPROCESSING" | grep -q "context_reformulation"; then
    echo "   ✅ Context reformulation applied!"
else
    echo "   ⚠️  Context reformulation NOT applied"
fi
echo "   Preprocessing steps: $PREPROCESSING"
echo ""

sleep 2

# Test 3: Follow-up query about custody
echo "================================================================================"
echo "TEST 3: Follow-up Query (Custody)"
echo "================================================================================"
echo ""

FOLLOWUP2='ماذا عن الحضانة في هذه القضية'

echo "Sending follow-up query: $FOLLOWUP2"
RESPONSE3=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"message\": \"$FOLLOWUP2\",
    \"conversation_id\": \"$CONVERSATION_ID\"
  }")

echo "Response:"
echo "$RESPONSE3" | python3 -m json.tool | head -50
echo ""

# Check for context reformulation
PREPROCESSING3=$(echo "$RESPONSE3" | python3 -c "import sys, json; print(json.load(sys.stdin).get('preprocessing_steps', []))" 2>/dev/null)

echo "✅ Follow-up query processed"
if echo "$PREPROCESSING3" | grep -q "context_reformulation"; then
    echo "   ✅ Context reformulation applied!"
else
    echo "   ⚠️  Context reformulation NOT applied"
fi
echo "   Preprocessing steps: $PREPROCESSING3"
echo ""

echo "================================================================================"
echo "TEST COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  - Conversation ID: $CONVERSATION_ID"
echo "  - Total queries: 3 (1 initial + 2 follow-ups)"
echo "  - Check Docker logs for conversation storage details"
echo ""

