# WhatsApp Business API Integration Guide

## 📋 Overview

This guide explains how to integrate WhatsApp Business API with your AI assistant (RAG system) to provide automated customer support via WhatsApp messaging.

**What You'll Build:**
- AI-powered WhatsApp chatbot
- Automatic message processing and responses
- RAG (Retrieval-Augmented Generation) integration
- Webhook-based real-time messaging

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  WhatsApp User                          │
│              (Sends message via WhatsApp)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Meta WhatsApp Business API                    │
│         (Receives message, sends webhook)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ POST /whatsapp/webhook
┌─────────────────────────────────────────────────────────┐
│              Your FastAPI Application                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  1. Webhook Endpoint (/whatsapp/webhook)        │  │
│  │     - Receives WhatsApp messages                 │  │
│  │     - Parses message data                        │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│                     ▼                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  2. RAG Service (AI Processing)                  │  │
│  │     - Searches vector database                   │  │
│  │     - Generates AI response                      │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│                     ▼                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  3. WhatsApp Client (send_text_message)         │  │
│  │     - Formats response for WhatsApp              │  │
│  │     - Sends reply via WhatsApp API               │  │
│  └──────────────────┬───────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ POST to Meta API
┌─────────────────────────────────────────────────────────┐
│           Meta WhatsApp Business API                    │
│         (Delivers message to user)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  WhatsApp User                          │
│              (Receives AI response)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Prerequisites

### 1. Meta Business Account Setup
- **Meta Business Account**: Create at [business.facebook.com](https://business.facebook.com)
- **WhatsApp Business App**: Create in Meta for Developers
- **Phone Number**: Verified business phone number

### 2. Required Credentials
You'll need these from Meta:
- `WHATSAPP_ACCESS_TOKEN` - API access token
- `WHATSAPP_PHONE_NUMBER_ID` - Your WhatsApp phone number ID
- `WHATSAPP_VERIFY_TOKEN` - Custom token for webhook verification (you create this)
- `WHATSAPP_BUSINESS_ACCOUNT_ID` - Your business account ID
- `WHATSAPP_APP_ID` - Your WhatsApp app ID
- `WHATSAPP_APP_SECRET` - Your app secret

---

## 🚀 Step-by-Step Integration

### Step 1: Get WhatsApp Business API Credentials

#### 1.1 Create Meta Developer Account
1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Click "Get Started" and create a developer account
3. Complete business verification (required for production)

#### 1.2 Create WhatsApp Business App
1. In Meta for Developers, click "Create App"
2. Select "Business" as app type
3. Fill in app details and create
4. Add "WhatsApp" product to your app

#### 1.3 Get Your Credentials
Navigate to WhatsApp > API Setup in your app dashboard:

```bash
# You'll find these values:
WHATSAPP_PHONE_NUMBER_ID=719099434628554  # From "Phone number ID"
WHATSAPP_BUSINESS_ACCOUNT_ID=790985490033754  # From "Business Account ID"
WHATSAPP_APP_ID=3241660855984391  # From app settings
```

#### 1.4 Generate Access Token
1. Go to WhatsApp > API Setup
2. Click "Generate Token" (temporary) or create a System User token (permanent)
3. Copy the access token:
```bash
WHATSAPP_ACCESS_TOKEN=EAAuERfvNoQcBPBMMCGM3EJtgsXBg1w6ODkWzhf2Iqvw3yZAfy7Voq78FCBj0vafjIznhqPOkZCuE3bdYwAAOdb7kj35K2ZBT6LMjrZCBGumVCkgCC0InckmcEcw9Fy8kKQRCZBllrUvH9MKLpWz5JykEBdjsO7ZAe8RNxQZAp6y7OYtCZAjZBjIIHtzrc25SSBOc7V1FzTgZC4C9Qvs3ptgj2ZCEhx2CgSBCFVklVjZAJsb38iBlBz1RrtWYjfChbfmVCM8ZD
```

#### 1.5 Create Verify Token
Create a random string for webhook verification:
```bash
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token_2024
```

---

### Step 2: Configure Your Application

#### 2.1 Add Environment Variables
Add these to your `.env` or `.env.prod` file:

```bash
# WhatsApp Business API Integration
WHATSAPP_ACCESS_TOKEN=your_access_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
WHATSAPP_APP_ID=your_app_id
WHATSAPP_APP_SECRET=your_app_secret
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token_2024
```

#### 2.2 Update Configuration (config.py)
The configuration is already set up in `app/config.py`:

```python
class Settings(BaseSettings):
    # WhatsApp Business API Configuration
    whatsapp_access_token: Optional[str] = Field(None, alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_phone_number_id: Optional[str] = Field(None, alias="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_verify_token: Optional[str] = Field(None, alias="WHATSAPP_VERIFY_TOKEN")
    whatsapp_business_account_id: Optional[str] = Field(None, alias="WHATSAPP_BUSINESS_ACCOUNT_ID")
    whatsapp_app_id: Optional[str] = Field(None, alias="WHATSAPP_APP_ID")
    whatsapp_app_secret: Optional[str] = Field(None, alias="WHATSAPP_APP_SECRET")
    
    def is_whatsapp_configured(self) -> bool:
        """Check if WhatsApp integration is properly configured."""
        return bool(
            self.whatsapp_access_token and
            self.whatsapp_phone_number_id and
            self.whatsapp_verify_token
        )
```

---

### Step 3: Set Up Webhook

#### 3.1 Make Your Server Publicly Accessible
Your webhook URL must be publicly accessible with HTTPS. Options:

**Option A: Production Server (Recommended)**
```bash
# Your webhook URL will be:
https://your-domain.com/whatsapp/webhook
```

**Option B: Development with ngrok**
```bash
# Install ngrok
brew install ngrok  # macOS
# or download from ngrok.com

# Start ngrok tunnel
ngrok http 8000

# Your webhook URL will be:
https://abc123.ngrok.io/whatsapp/webhook
```

#### 3.2 Configure Webhook in Meta
1. Go to WhatsApp > Configuration in your app dashboard
2. Click "Edit" next to Webhook
3. Enter your webhook URL:
   ```
   https://your-domain.com/whatsapp/webhook
   ```
4. Enter your verify token (same as `WHATSAPP_VERIFY_TOKEN`)
5. Click "Verify and Save"

#### 3.3 Subscribe to Webhook Events
After verification, subscribe to these events:
- ✅ `messages` - Receive incoming messages
- ✅ `message_status` - Track message delivery (optional)

---

### Step 4: Test the Integration

#### 4.1 Check Configuration Status
```bash
curl https://your-domain.com/whatsapp/status
```

Expected response:
```json
{
  "status": "configured",
  "configured": true,
  "health": {
    "status": "healthy",
    "configured": true
  },
  "phone_number_id": "719099434628554",
  "webhook_url": "https://your-domain.com/whatsapp/webhook"
}
```

#### 4.2 Test with Simulation Endpoint
```bash
curl -X POST https://your-domain.com/whatsapp/simulate \
  -H "Content-Type: application/json" \
  -d '{
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
            "type": "text",
            "text": {
              "body": "Hello, what can you help me with?"
            }
          }]
        }
      }]
    }]
  }'
```

#### 4.3 Send Real WhatsApp Message
1. Add your test phone number in Meta dashboard (WhatsApp > API Setup > "To" field)
2. Send a message from your WhatsApp to the business number
3. Check logs to see the message processing

---

### Step 5: Monitor and Debug

#### 5.1 Check Debug Information
```bash
curl https://your-domain.com/whatsapp/debug
```

#### 5.2 View Application Logs
```bash
# Docker logs
docker logs saia-rag-api-prod --tail 100 -f

# Look for these log entries:
# - "WhatsApp webhook received"
# - "WhatsApp message parsed successfully"
# - "WhatsApp message sent successfully"
```

#### 5.3 Common Issues and Solutions

**Issue: Webhook verification fails**
```
Solution: Ensure WHATSAPP_VERIFY_TOKEN matches exactly in both:
- Your .env file
- Meta webhook configuration
```

**Issue: 401 Unauthorized errors**
```
Solution: Your access token may have expired
- Generate a new permanent token using System User
- Update WHATSAPP_ACCESS_TOKEN in .env
- Restart your application
```

**Issue: Messages not being received**
```
Solution: Check webhook subscription
- Verify webhook is subscribed to "messages" event
- Check webhook URL is correct and accessible
- Verify SSL certificate is valid
```

---

## 📚 API Endpoints Reference

### Webhook Endpoints

#### `GET /whatsapp/verify`
**Purpose**: Webhook verification (called by Meta during setup)

**Query Parameters**:
- `hub.mode` - Verification mode
- `hub.verify_token` - Your verify token
- `hub.challenge` - Challenge string from Meta

**Response**: Plain text challenge string

---

#### `POST /whatsapp/webhook`
**Purpose**: Receive incoming WhatsApp messages

**Request Body** (from Meta):
```json
{
  "object": "whatsapp_business_account",
  "entry": [{
    "id": "BUSINESS_ACCOUNT_ID",
    "changes": [{
      "value": {
        "messaging_product": "whatsapp",
        "metadata": {
          "display_phone_number": "+1234567890",
          "phone_number_id": "PHONE_NUMBER_ID"
        },
        "messages": [{
          "from": "1234567890",
          "id": "MESSAGE_ID",
          "timestamp": "1234567890",
          "type": "text",
          "text": {
            "body": "User message here"
          }
        }]
      }
    }]
  }]
}
```

**Response**: `{"status": "received"}`

---

### Utility Endpoints

#### `GET /whatsapp/status`
Check WhatsApp integration status and health

#### `GET /whatsapp/debug`
Get detailed debug information

#### `POST /whatsapp/simulate`
Test webhook processing without real WhatsApp messages

---

## 🔧 Code Implementation

### Key Files

1. **`app/whatsapp_client.py`** - WhatsApp API client
   - `send_text_message()` - Send messages
   - `parse_webhook_message()` - Parse incoming messages
   - `verify_webhook()` - Verify webhook setup

2. **`app/main.py`** - Webhook endpoints
   - `/whatsapp/verify` - Webhook verification
   - `/whatsapp/webhook` - Message receiver
   - `/whatsapp/status` - Status check

3. **`app/config.py`** - Configuration
   - WhatsApp credentials
   - `is_whatsapp_configured()` - Check setup

---

## 🎯 Message Flow Example

```python
# 1. User sends WhatsApp message
"What are the custody requirements in Saudi Arabia?"

# 2. Meta sends webhook to your server
POST /whatsapp/webhook
{
  "messages": [{
    "from": "966501234567",
    "text": {"body": "What are the custody requirements..."}
  }]
}

# 3. Your app processes with RAG
- Searches vector database for relevant legal documents
- Generates AI response using OpenAI
- Formats response for WhatsApp

# 4. Your app sends response via WhatsApp API
POST https://graph.facebook.com/v18.0/{phone_number_id}/messages
{
  "messaging_product": "whatsapp",
  "to": "966501234567",
  "type": "text",
  "text": {
    "body": "شروط الحضانة وفقاً للنظام السعودي تشمل..."
  }
}

# 5. User receives AI response in WhatsApp
```

---

## 🔐 Security Best Practices

1. **Use System User Tokens** (not temporary tokens)
2. **Verify webhook signatures** (implement in production)
3. **Use HTTPS only** for webhook URLs
4. **Rotate tokens regularly**
5. **Store credentials securely** (environment variables, not code)
6. **Implement rate limiting** (already included in middleware)
7. **Log security events** (already implemented with structlog)

---

## 📊 Production Checklist

- [ ] Business verification completed in Meta
- [ ] Permanent access token generated (System User)
- [ ] Webhook URL is HTTPS with valid SSL certificate
- [ ] Webhook verified and subscribed to events
- [ ] Environment variables configured
- [ ] Application restarted with new config
- [ ] Test messages sent and received successfully
- [ ] Monitoring and logging set up
- [ ] Error handling tested
- [ ] Rate limiting configured

---

## 🆘 Support Resources

- **Meta WhatsApp Business API Docs**: https://developers.facebook.com/docs/whatsapp
- **WhatsApp Business Platform**: https://business.whatsapp.com
- **Meta for Developers**: https://developers.facebook.com
- **API Reference**: https://developers.facebook.com/docs/whatsapp/cloud-api/reference

---

## 📞 Testing Your Integration

### Quick Test Script
```bash
# 1. Check status
curl https://your-domain.com/whatsapp/status

# 2. Send test message from your WhatsApp

# 3. Check logs
docker logs saia-rag-api-prod --tail 50 | grep -i whatsapp

# 4. Verify response received in WhatsApp
```

---

**🎉 Congratulations!** You now have a fully functional WhatsApp AI assistant integrated with your RAG system!

