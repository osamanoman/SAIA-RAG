# How to Generate a Permanent WhatsApp Access Token (Never Expires)

**Source:** Meta Business Manager - System User Access Tokens  
**Documentation:** https://business.facebook.com/

---

## 📋 **Complete Step-by-Step Guide**

### **Step 1: Access Meta Business Suite**
1. Go to: **https://business.facebook.com/**
2. Log in with your Meta account
3. Select your **Business Portfolio**

---

### **Step 2: Navigate to Business Settings**
1. Click **Business Settings** in the left menu (gear icon ⚙️)
2. You'll see various sections: Users, Accounts, Data Sources, etc.

---

### **Step 3: Create a System User** 👤

1. In the left sidebar, under **Users** section, click **System Users**
2. Click the **Add** button (top right)
3. Fill in the details:
   - **System User Name**: `WhatsApp API Bot` (or any name you prefer)
   - **Role**: Select **Admin** (required for full access)
4. Click **Create System User**

**✅ System User Created!**

---

### **Step 4: Assign Assets to the System User** 📱

1. Click on the **System User** you just created
2. Click **Add Assets** button
3. In the **Select Asset Type** dropdown:
   - Choose **Apps**
4. Find and select your **WhatsApp App** (App ID: `3241660855984391`)
5. Grant **Full Control** permission
6. Click **Save Changes**

**Alternative: Assign WhatsApp Accounts**
- You can also assign **WhatsApp Accounts** (WABAs) directly
- Go to **Add Assets** → Select **WhatsApp Accounts**
- Select your WABA and grant **Full Control**

---

### **Step 5: Generate the Permanent Access Token** 🔑

1. With the **System User** still selected, click **Generate New Token**
2. A dialog will appear:
   - **App**: Select your app (`3241660855984391` or your active app)
   - **Token Expiration**: Select **Never** (60 days or Never)
   - **Available Permissions**: Check these boxes:
     - ☑️ `whatsapp_business_messaging`
     - ☑️ `whatsapp_business_management`
     - ☑️ `business_management` (optional, for advanced features)
3. Click **Generate Token**

**🎉 Token Generated!**

---

### **Step 6: Securely Store the Token** 🔒

1. **Copy the token** (starts with `EAA...`)
2. **⚠️ IMPORTANT**: This token will **NOT** be shown again!
3. Store it securely in:
   - Your `.env` file
   - A password manager
   - A secure vault

---

## 📝 **What You'll Get**

```
Token Format: EAA...
Token Type: SYSTEM_USER
Expires: Never (0 timestamp)
Permissions:
  - whatsapp_business_messaging
  - whatsapp_business_management
```

---

## ✅ **Verification**

To verify your permanent token:

```bash
curl 'https://graph.facebook.com/v21.0/debug_token?input_token=YOUR_TOKEN' \
-H 'Authorization: Bearer YOUR_TOKEN'
```

**Expected Response:**
```json
{
  "data": {
    "type": "SYSTEM_USER",
    "expires_at": 0,  ← Never expires!
    "is_valid": true,
    "scopes": [
      "whatsapp_business_messaging",
      "whatsapp_business_management"
    ]
  }
}
```

---

## 🚀 **Using the Token**

Once you have the permanent token, update your environment:

```bash
# .env file
WHATSAPP_ACCESS_TOKEN=EAA...your_permanent_token_here
```

Restart your application to apply the changes.

---

## ⚠️ **Important Notes**

1. **System User tokens NEVER expire** (unless revoked manually)
2. **Keep it secure** - treat it like a password
3. **Don't share** the token publicly or commit it to git
4. **Admin role** is required to generate permanent tokens
5. **One token per system user** - you can create multiple system users if needed

---

## 🔄 **Token vs System User Comparison**

| Feature | Temporary Token | System User Token |
|---------|----------------|-------------------|
| **Expires** | 24 hours - 60 days | Never |
| **Requires Login** | Yes | No |
| **Production Use** | ❌ Not recommended | ✅ Recommended |
| **Renewal Needed** | Yes | No |
| **Permissions** | Limited | Full control |

---

## 📚 **Additional Resources**

- **Meta Business Manager**: https://business.facebook.com/
- **System Users Guide**: https://www.facebook.com/business/help/503306463479099
- **WhatsApp Business API Docs**: https://developers.facebook.com/docs/whatsapp
- **Graph API Explorer**: https://developers.facebook.com/tools/explorer/

---

## 🆘 **Troubleshooting**

### Problem: "Cannot create system user"
**Solution**: Make sure you have Admin role in the Business Portfolio

### Problem: "App not showing in dropdown"
**Solution**: Add the app to your Business Portfolio first

### Problem: "Permissions not available"
**Solution**: Make sure WhatsApp product is added to your app

### Problem: "Token expires immediately"
**Solution**: Use System User tokens, not App Access Tokens

---

**Last Updated:** November 17, 2025  
**Meta API Version:** v21.0+


