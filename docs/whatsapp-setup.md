# WhatsApp Business API Setup Guide

Complete guide for setting up WhatsApp Business API integration with SAIA-RAG.

## 🎯 **Overview**

SAIA-RAG integrates with WhatsApp Business API to provide AI-powered customer support via WhatsApp messaging. This integration allows customers to:

- Send questions via WhatsApp
- Receive AI-generated responses based on your knowledge base
- Get consistent support across all channels (Web UI, WhatsApp, API)

## 🔧 **Prerequisites**

### **1. Meta Developer Account**
- [Meta for Developers](https://developers.facebook.com/) account
- Verified business account
- WhatsApp Business API access

### **2. WhatsApp Business Account**
- Active WhatsApp Business account
- Phone number for business
- Business verification completed

### **3. Application Setup**
- Facebook App created
- WhatsApp Business API product added
- Webhook configured

## 📋 **Step-by-Step Setup**

### **Step 1: Create Facebook App**

1. Go to [Meta for Developers](https://developers.facebook.com/)
2. Click "Create App"
3. Select "Business" as app type
4. Fill in app details and create

### **Step 2: Add WhatsApp Business API**

1. In your app dashboard, click "Add Product"
2. Find "WhatsApp" and click "Set Up"
3. Complete the setup wizard

### **Step 3: Configure Webhook**

1. In WhatsApp Business API settings, click "Configure Webhook"
2. Set webhook URL: `https://yourdomain.com/whatsapp/webhook`
3. Set verify token: Choose a secure random string
4. Subscribe to messages and message_deliveries events

### **Step 4: Get API Credentials**

1. **Access Token**: Copy from WhatsApp Business API settings
2. **Phone Number ID**: Found in WhatsApp Business API dashboard
3. **Verify Token**: The token you set in webhook configuration
4. **Business Account ID**: Your WhatsApp Business account ID

## 🔐 **Environment Configuration**

### **1. Create Production Environment File**

Copy the example environment file:
```bash
cp env.prod.example .env.prod
```

### **2. Configure WhatsApp Settings**

Edit `.env.prod` and add your WhatsApp credentials:
```bash
# === WHATSAPP BUSINESS API CONFIGURATION ===
WHATSAPP_ACCESS_TOKEN=your-actual-access-token-here
WHATSAPP_PHONE_NUMBER_ID=your-actual-phone-number-id-here
WHATSAPP_VERIFY_TOKEN=your-actual-verify-token-here
WHATSAPP_BUSINESS_ACCOUNT_ID=your-business-account-id-here
WHATSAPP_APP_ID=your-app-id-here
WHATSAPP_APP_SECRET=your-app-secret-here
```

### **3. Update Base URL**

Set your actual domain:
```bash
BASE_URL=https://yourdomain.com
```

## 🧪 **Testing the Integration**

### **1. Test Configuration**

Check if WhatsApp is properly configured:
```bash
curl http://yourdomain.com/whatsapp/status
```

Expected response:
```json
{
  "status": "configured",
  "configured": true,
  "health": {
    "status": "healthy",
    "configured": true,
    "phone_number_id": "your_phone_id",
    "api_version": "v18.0"
  }
}
```

### **2. Test Webhook Verification**

Test the verification endpoint:
```bash
curl "http://yourdomain.com/whatsapp/verify?hub.mode=subscribe&hub.verify_token=your_token&hub.challenge=test_challenge"
```

### **3. Simulate Webhook Message**

Test message processing without actual WhatsApp:
```bash
curl -X POST http://yourdomain.com/whatsapp/simulate \
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
            "text": {
              "body": "What services do you offer?"
            },
            "type": "text"
          }]
        },
        "field": "messages"
      }]
    }]
  }'
```

### **4. Test Debug Information**

Get comprehensive debug info:
```bash
curl http://yourdomain.com/whatsapp/debug
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. "WhatsApp not configured" Error**

**Cause**: Missing environment variables
**Solution**: Check `.env.prod` file and ensure all WhatsApp variables are set

#### **2. Webhook Verification Fails**

**Cause**: Incorrect verify token
**Solution**: Ensure `WHATSAPP_VERIFY_TOKEN` matches what you set in Meta dashboard

#### **3. API Connectivity Issues**

**Cause**: Invalid access token or phone number ID
**Solution**: Verify credentials in Meta dashboard and regenerate if needed

#### **4. Message Processing Fails**

**Cause**: Webhook payload format mismatch
**Solution**: Use the simulation endpoint to test with correct format

### **Debug Steps**

1. **Check Configuration**:
   ```bash
   curl http://yourdomain.com/whatsapp/debug
   ```

2. **Check Health Status**:
   ```bash
   curl http://yourdomain.com/whatsapp/status
   ```

3. **Test Endpoint Routing**:
   ```bash
   curl http://yourdomain.com/whatsapp/test
   ```

4. **Check Application Logs**:
   ```bash
   docker logs saia-rag-api-prod
   ```

## 📱 **WhatsApp Message Flow**

### **1. Customer Sends Message**
- Customer types message in WhatsApp
- WhatsApp sends webhook to your server

### **2. Webhook Processing**
- Server receives webhook (responds within 80ms)
- Message parsed and validated
- Background task scheduled for processing

### **3. RAG Processing**
- Message processed through unified RAG pipeline
- Same processing as Web UI for consistency
- Response generated with context from knowledge base

### **4. Response Delivery**
- AI response sent back via WhatsApp API
- Customer receives response in WhatsApp

## 🔒 **Security Considerations**

### **1. Webhook Verification**
- Always verify webhook authenticity
- Use secure verify tokens
- Validate webhook payload structure

### **2. API Key Protection**
- Keep access tokens secure
- Use environment variables
- Never commit credentials to version control

### **3. Rate Limiting**
- Implement rate limiting for webhook endpoints
- Monitor API usage
- Respect WhatsApp API limits

## 📊 **Monitoring & Analytics**

### **1. Health Checks**
- Regular health check monitoring
- API connectivity verification
- Webhook delivery status

### **2. Performance Metrics**
- Response time monitoring
- Success/failure rates
- API usage tracking

### **3. Error Tracking**
- Comprehensive error logging
- Failed message tracking
- Webhook delivery failures

## 🚀 **Production Deployment**

### **1. Environment Setup**
```bash
# Copy environment template
cp env.prod.example .env.prod

# Edit with your credentials
nano .env.prod

# Deploy to production
./deploy.sh your-server-ip root
```

### **2. Webhook Configuration**
- Set webhook URL in Meta dashboard
- Configure verify token
- Subscribe to required events

### **3. Testing**
- Test verification endpoint
- Test message simulation
- Verify end-to-end flow

### **4. Monitoring**
- Set up health check monitoring
- Configure error alerting
- Monitor API usage

## 🎉 **Success Indicators**

Your WhatsApp integration is working correctly when:

✅ **Status endpoint** returns "configured: true"  
✅ **Health check** returns "status: healthy"  
✅ **Webhook verification** returns challenge string  
✅ **Message simulation** processes successfully  
✅ **Real WhatsApp messages** are processed and responded to  

## 📞 **Support**

If you encounter issues:

1. Check the debug endpoint: `/whatsapp/debug`
2. Review application logs
3. Test with simulation endpoint
4. Verify Meta dashboard configuration
5. Check environment variables

The WhatsApp integration provides a seamless way to extend your AI-powered customer support to WhatsApp users while maintaining consistency with your web interface responses.

