# WhatsApp Session Management Implementation - Best Practices

## Overview
Complete redesign of conversation/session management based on industry best practices for WhatsApp chatbots.

---

## Problem Statement

### Current Issues
1. **No Session Expiry:** Conversations never reset, context grows forever
2. **Poor Case Extraction:** Only keyword matching, no AI-powered entity extraction
3. **No Reset Mechanism:** Users can't start fresh conversations
4. **Lost Case Context:** When user says "I mean Mariam's case", bot has no idea what that is
5. **Generic Responses:** Bot gives general law instead of case-specific answers

### Example of Problem
```
User: [Full case: Muhammad & Reem, 2 kids ages 6 & 4, separated 10 months...]
Bot: [Legal answer]
User: "لم افهم نفقة الاطفال" (I didn't understand children's alimony)
Bot: [Generic alimony law] ❌ Should be: Case-specific for Muhammad/Reem's kids!
User: "اقصد لقضؤة مريم" (I mean Mariam's case) 
Bot: "No information found" ❌ Bot doesn't know who Mariam is!
```

---

## Solution Architecture

### 1. Session Structure (Per Phone Number)

```python
{
  "session_id": "whatsapp:+966500000000:+966558807769",
  "user_phone": "+966558807769",
  "bot_number": "+966500000000",
  
  # Session Management
  "topic_id": "topic-1732012345",
  "created_at": "2025-11-18T09:00:00Z",
  "last_interaction_at": "2025-11-18T09:42:00Z",
  "state": "active",  # active, idle, expired
  
  # User Profile
  "language_preference": "ar",
  "timezone": "Asia/Riyadh",
  
  # Rich Case Facts (AI-extracted)
  "case_facts": {
    "case_title": "المطالبة بالنفقة وحضانة الأطفال",
    "parties": {
      "plaintiff": {
        "name": "ريم",
        "role": "wife",
        "represented_by": "محامٍ"
      },
      "defendant": {
        "name": "محمد",
        "role": "husband"
      },
      "affected": [
        {"name": "طفل", "age": 6, "relation": "son"},
        {"name": "طفل", "age": 4, "relation": "son"}
      ]
    },
    "key_facts": {
      "marriage_duration": "8 years",
      "separation_duration": "10 months",
      "marital_status": "separated_no_divorce",
      "custody_status": "disputed",
      "financial_support": "none_for_10_months"
    },
    "legal_issues": [
      "child_alimony",
      "custody",
      "housing_costs",
      "visitation_rights"
    ],
    "plaintiff_requests": [
      "إلزام الزوج بالنفقة الشهرية للأطفال",
      "إثبات حضانة الأطفال للأم",
      "الزام الزوج بمصاريف إيجار السكن",
      "طلب زيارة الأب للأطفال في مركز الزيارة"
    ]
  },
  
  # Conversation History (last 15 messages)
  "messages": [
    {
      "role": "user",
      "content": "...",
      "timestamp": "2025-11-18T09:00:00Z"
    },
    {
      "role": "assistant",
      "content": "...",
      "timestamp": "2025-11-18T09:00:15Z"
    }
  ]
}
```

---

## 2. Session Lifecycle Rules

### Rule 1: Time-Based Expiry
```python
SESSION_IDLE_TIMEOUT = 45  # minutes

if (now - last_interaction_at) > SESSION_IDLE_TIMEOUT:
    # Create new session
    topic_id = generate_new_topic_id()
    case_facts = {}
    messages = []
    
    # Inform user
    message = """
    تم فتح محادثة جديدة بعد فترة من عدم النشاط. 
    إذا كنت تريد مناقشة نفس القضية، أخبرني وسأسترجع التفاصيل.
    
    New conversation started after period of inactivity.
    If you want to discuss the same case, let me know.
    """
```

### Rule 2: Explicit Reset Commands
```python
RESET_COMMANDS = [
    "ابدأ من جديد",
    "ابدا من جديد",
    "محادثة جديدة",
    "جديد",
    "/reset",
    "/new",
    "new topic",
    "forget everything"
]

if message.strip().lower() in RESET_COMMANDS:
    # Hard reset
    create_new_session()
    reply = "تم! بدأنا محادثة جديدة. كيف يمكنني مساعدتك؟"
```

### Rule 3: Context Window (Last 15 messages)
```python
# Always keep full history in DB
# But when calling LLM, only use last 15 message pairs
context_messages = messages[-15:]
```

---

## 3. AI-Powered Case Fact Extraction

### Current (Bad):
```python
# Just keyword matching!
if "حضانة" in query:
    key_issues.append("custody")
```

### New (AI-Powered):
```python
async def extract_case_facts_ai(message: str) -> Dict[str, Any]:
    """
    Use GPT-4o-mini to extract structured case facts from user message.
    """
    extraction_prompt = """
    Extract structured legal case information from this Arabic text.
    Focus on:
    - Parties: names, roles (plaintiff/defendant/wife/husband)
    - Key facts: ages, durations, status
    - Legal issues: custody, alimony, divorce, visitation
    - Specific requests/demands
    
    Respond in JSON:
    {
      "case_title": "...",
      "parties": {...},
      "key_facts": {...},
      "legal_issues": [...],
      "requests": [...]
    }
    
    Text: {message}
    """
    
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a legal document analyzer for Saudi law."},
            {"role": "user", "content": extraction_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    return json.loads(response.choices[0].message.content)
```

---

## 4. Context-Aware Query Reformulation

### When user says: "لم افهم نفقة الاطفال"

**Old behavior:**
```python
# Just adds first query
reformulated = f"{query}. السياق: {first_query}"
# Result: Generic query, no case details
```

**New behavior:**
```python
# Uses rich case facts!
reformulated = f"""
السؤال: {query}

سياق القضية:
- الأطراف: محمد (الزوج) و ريم (الزوجة)
- الأطفال: طفلان عمر 6 سنوات و 4 سنوات
- الوضع: انفصال بدون طلاق منذ 10 أشهر
- المشكلة: عدم إنفاق على الأطفال منذ 10 أشهر
- المطالب: نفقة شهرية للأطفال، حضانة، إيجار سكن، زيارة

يرجى توضيح نفقة الأطفال في سياق هذه القضية بالتحديد.
"""
# Result: Case-specific, detailed context!
```

### When user says: "اقصد لقضؤة مريم"

**Old behavior:**
```python
# No idea who "Mariam" is
# Searches for "Mariam case" in legal docs
# Finds nothing → "No information found"
```

**New behavior:**
```python
# Checks case_facts for party names
if "ريم" in case_facts["parties"]["plaintiff"]["name"]:
    # User might be referring to plaintiff (typo: مريم vs ريم)
    reformulated = f"""
    يبدو أنك تقصد قضية ريم (المدعية) ضد محمد (المدعى عليه).
    
    تفاصيل القضية:
    {format_case_facts(case_facts)}
    
    ما الذي تريد معرفته بالتحديد؟
    """
```

---

## 5. Session Reset Detection

### Semantic Detection (AI-powered)
```python
async def detect_new_topic_intent(query: str, case_facts: Dict) -> bool:
    """
    Detect if user wants to discuss a completely different topic.
    """
    if not case_facts:
        return False
    
    detection_prompt = f"""
    Current case is about: {case_facts.get('legal_issues', [])}
    
    User says: "{query}"
    
    Is this:
    A) Same case (follow-up, clarification, related question)
    B) New topic (different case, unrelated legal question)
    
    Respond: A or B
    """
    
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": detection_prompt}],
        temperature=0.0,
        max_tokens=5
    )
    
    return response.choices[0].message.content.strip() == "B"
```

---

## 6. Implementation Steps

### Step 1: Enhance ConversationContext Model
```python
@dataclass
class ConversationContext:
    conversation_id: str
    topic_id: str
    created_at: datetime
    last_interaction_at: datetime
    state: ConversationState
    
    # NEW: Rich case facts
    case_facts: Dict[str, Any] = field(default_factory=dict)
    
    # NEW: Session metadata
    idle_timeout_minutes: int = 45
    auto_reset_enabled: bool = True
```

### Step 2: Add Session Expiry Check
```python
async def check_session_expiry(conversation_id: str) -> bool:
    """
    Check if session has expired (idle > 45 min).
    Returns True if new session needed.
    """
    conversation = await get_conversation(conversation_id)
    if not conversation:
        return False
    
    idle_minutes = (datetime.utcnow() - conversation.last_interaction_at).total_seconds() / 60
    
    if idle_minutes > conversation.idle_timeout_minutes:
        logger.info(
            "Session expired due to inactivity",
            conversation_id=conversation_id,
            idle_minutes=idle_minutes
        )
        return True
    
    return False
```

### Step 3: Implement AI Case Fact Extraction
```python
async def extract_case_facts(message: str) -> Dict[str, Any]:
    """
    AI-powered case fact extraction.
    """
    # Only extract if message is long enough (likely case description)
    if len(message.split()) < 20:
        return {}
    
    extraction_prompt = """..."""  # See section 3
    
    # Call GPT-4o-mini
    response = await openai.chat.completions.create(...)
    
    return json.loads(response.choices[0].message.content)
```

### Step 4: Detect Reset Commands
```python
def is_reset_command(message: str) -> bool:
    """
    Check if user wants to reset/start new session.
    """
    message_lower = message.strip().lower()
    
    reset_phrases = [
        "ابدأ من جديد", "ابدا من جديد",
        "محادثة جديدة", "جديد",
        "/reset", "/new",
        "new topic", "forget"
    ]
    
    return any(phrase in message_lower for phrase in reset_phrases)
```

### Step 5: Update WhatsApp Handler
```python
async def handle_whatsapp_message(user_phone, message_text):
    conversation_id = f"whatsapp_{user_phone}"
    
    # Step 1: Check for reset command
    if is_reset_command(message_text):
        await reset_session(conversation_id)
        return "تم! بدأنا محادثة جديدة. كيف يمكنني مساعدتك؟"
    
    # Step 2: Check session expiry
    if await check_session_expiry(conversation_id):
        await create_new_session(conversation_id)
        # Optionally inform user
    
    # Step 3: Extract case facts (if first message or case description)
    conversation = await get_conversation(conversation_id)
    if not conversation.case_facts and len(message_text.split()) > 20:
        case_facts = await extract_case_facts(message_text)
        await update_case_facts(conversation_id, case_facts)
    
    # Step 4: Continue with RAG processing
    rag_response = await process_chat_request_whatsapp(
        query=message_text,
        conversation_id=conversation_id
    )
    
    return rag_response
```

---

## 7. Query Reformulation with Rich Context

```python
async def reformulate_with_rich_context(
    query: str,
    case_facts: Dict[str, Any]
) -> str:
    """
    Reformulate query using rich case facts.
    """
    if not case_facts:
        return query
    
    # Build context from case facts
    context_lines = []
    
    if "parties" in case_facts:
        context_lines.append(f"الأطراف: {format_parties(case_facts['parties'])}")
    
    if "key_facts" in case_facts:
        context_lines.append(f"الوقائع: {format_key_facts(case_facts['key_facts'])}")
    
    if "legal_issues" in case_facts:
        issues = ", ".join(case_facts["legal_issues"])
        context_lines.append(f"القضايا القانونية: {issues}")
    
    context_str = "\n".join(context_lines)
    
    reformulation_prompt = f"""
    السؤال الأصلي: {query}
    
    سياق القضية:
    {context_str}
    
    أعد صياغة السؤال ليكون واضحاً ومحدداً في سياق هذه القضية.
    إذا كان السؤال يشير إلى "القضية" أو "الحالة"، استخدم التفاصيل المذكورة أعلاه.
    """
    
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت مساعد قانوني يعيد صياغة الأسئلة لتكون واضحة ومحددة."},
            {"role": "user", "content": reformulation_prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()
```

---

## 8. User Experience Improvements

### Scenario 1: Session Expired
```
[User hasn't messaged for 60 minutes]

User: "ما الإجراءات التالية؟"

Bot: """
مرحباً! لاحظت أنه مر وقت منذ آخر محادثة (60 دقيقة).

إذا كنت تريد مناقشة نفس القضية (محمد و ريم - النفقة والحضانة)، 
يمكنني مساعدتك مباشرة.

أو إذا كان لديك موضوع جديد، أخبرني به.
"""
```

### Scenario 2: Explicit Reset
```
User: "ابدأ من جديد"

Bot: """
تم! ✅ بدأنا محادثة جديدة من الصفر.

أنا مساعدك في قانون الأحوال الشخصية السعودي.
كيف يمكنني مساعدتك اليوم؟
"""
```

### Scenario 3: Reference to Case
```
User: "اقصد لقضؤة مريم"

Bot: """
أعتقد أنك تقصد قضية *ريم* (المدعية) ضد محمد.

📋 ملخص القضية:
• الأطراف: ريم (زوجة) vs محمد (زوج)
• الأطفال: طفلان (6 و 4 سنوات)
• الوضع: انفصال بدون طلاق منذ 10 أشهر
• المطالب: نفقة، حضانة، إيجار، زيارة

ما الذي تريد معرفته عن هذه القضية؟
"""
```

---

## 9. Testing Scenarios

### Test 1: Time-Based Expiry
1. Send case details
2. Wait 46 minutes
3. Send "ما الإجراءات؟"
4. ✅ Should inform about new session and ask for context

### Test 2: Explicit Reset
1. Send case details
2. Bot responds
3. Send "ابدأ من جديد"
4. ✅ Should reset and confirm

### Test 3: Case Reference
1. Send case with names (Muhammad & Reem)
2. Ask "لم افهم نفقة الاطفال"
3. ✅ Should explain alimony IN CONTEXT of Muhammad/Reem's case
4. Ask "اقصد لقضؤة مريم"
5. ✅ Should recognize typo/reference to Reem and provide case summary

### Test 4: Follow-Up After Expiry
1. Send case details
2. Wait 46 minutes (expired)
3. Send "اكمل"
4. ✅ Should inform about expiry and ask what case to discuss

---

## 10. Implementation Priority

### Phase 1: Core Session Management (Critical)
- [ ] Add `last_interaction_at` to ConversationContext
- [ ] Implement `check_session_expiry()`
- [ ] Add session expiry notification message
- [ ] Test time-based expiry

### Phase 2: Reset Commands (High Priority)
- [ ] Implement `is_reset_command()`
- [ ] Add reset handler in WhatsApp webhook
- [ ] Test explicit reset commands

### Phase 3: AI Case Extraction (High Priority)
- [ ] Implement `extract_case_facts_ai()`
- [ ] Add `case_facts` field to ConversationContext
- [ ] Extract facts from first long message
- [ ] Test case fact extraction

### Phase 4: Rich Context Reformulation (Critical for UX)
- [ ] Implement `reformulate_with_rich_context()`
- [ ] Use case facts in reformulation
- [ ] Test case-specific query reformulation

### Phase 5: Semantic Topic Detection (Optional)
- [ ] Implement `detect_new_topic_intent()`
- [ ] Auto-detect topic changes
- [ ] Ask user for confirmation before reset

---

## 11. Code Files to Modify

1. **`app/conversation_memory.py`**
   - Add `case_facts: Dict` field to `ConversationContext`
   - Add `last_interaction_at: datetime` tracking
   - Implement `check_session_expiry()`
   - Implement `reset_session()`

2. **`app/query_processor.py`**
   - Replace `_extract_case_facts()` with AI-powered version
   - Enhance `_reformulate_with_context()` to use rich facts
   - Add `is_reset_command()` detection

3. **`app/main.py`**
   - Add session expiry check in WhatsApp handler
   - Add reset command detection
   - Add case fact extraction on first message

4. **`app/config.py`**
   - Add `SESSION_IDLE_TIMEOUT_MINUTES = 45`
   - Add `ENABLE_AI_CASE_EXTRACTION = True`

---

## 12. Deployment Checklist

- [ ] Code changes implemented and tested locally
- [ ] Session expiry tested (with reduced timeout for testing)
- [ ] Reset commands tested
- [ ] AI case extraction tested
- [ ] Rich context reformulation tested
- [ ] Deployed to production
- [ ] Monitored WhatsApp conversations for 24 hours
- [ ] User feedback collected

---

## Conclusion

This implementation brings SAIA-RAG in line with industry best practices for WhatsApp chatbots:

✅ **Proper session management** (time-based expiry)
✅ **Rich case context** (AI-extracted facts, not just keywords)
✅ **User control** (reset commands)
✅ **Case-aware responses** (understands "I mean Mariam's case")
✅ **Context window management** (last 15 messages)
✅ **Professional UX** (clear session transitions)

**Result:** Users can have natural, context-aware conversations about their specific legal cases, with proper session boundaries and the ability to start fresh when needed.

