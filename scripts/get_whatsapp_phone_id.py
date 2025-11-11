#!/usr/bin/env python3
"""
Script to fetch WhatsApp Phone Number ID from Meta Graph API.

Usage:
    python scripts/get_whatsapp_phone_id.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_whatsapp_phone_numbers():
    """Fetch all WhatsApp phone numbers associated with the business account."""
    
    # Get credentials from environment
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    business_account_id = os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID")
    
    if not access_token:
        print("❌ Error: WHATSAPP_ACCESS_TOKEN not found in .env file")
        sys.exit(1)
    
    if not business_account_id:
        print("❌ Error: WHATSAPP_BUSINESS_ACCOUNT_ID not found in .env file")
        sys.exit(1)
    
    print(f"🔍 Fetching phone numbers for Business Account ID: {business_account_id}")
    print(f"🔑 Using access token: {access_token[:20]}...")
    print()
    
    # Meta Graph API endpoint
    url = f"https://graph.facebook.com/v18.0/{business_account_id}/phone_numbers"
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        print("📡 Making API request to Meta Graph API...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "data" not in data or len(data["data"]) == 0:
            print("⚠️  No phone numbers found for this business account")
            print("\nPossible reasons:")
            print("1. Phone number not yet added to the business account")
            print("2. Access token doesn't have permission to read phone numbers")
            print("3. Business account ID is incorrect")
            return
        
        print("✅ Successfully retrieved phone numbers!\n")
        print("=" * 70)
        
        for idx, phone in enumerate(data["data"], 1):
            print(f"\n📱 Phone Number {idx}:")
            print(f"   Phone Number ID: {phone.get('id', 'N/A')}")
            print(f"   Display Name: {phone.get('display_phone_number', 'N/A')}")
            print(f"   Verified Name: {phone.get('verified_name', 'N/A')}")
            print(f"   Quality Rating: {phone.get('quality_rating', 'N/A')}")
            print(f"   Status: {phone.get('account_mode', 'N/A')}")
            
            # Check if this is the current phone number
            current_phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
            if phone.get('id') == current_phone_id:
                print(f"   ✅ This is your CURRENT phone number in .env")
            else:
                print(f"   ⚠️  This is NOT your current phone number in .env")
        
        print("\n" + "=" * 70)
        print("\n📝 To update your .env file:")
        print("1. Copy the Phone Number ID from above")
        print("2. Update WHATSAPP_PHONE_NUMBER_ID in your .env file")
        print("3. Restart your application")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        
        if e.response.status_code == 401:
            print("\n⚠️  Authentication failed. Possible reasons:")
            print("1. Access token has expired")
            print("2. Access token doesn't have required permissions")
            print("3. Access token is invalid")
            print("\n💡 Solution: Generate a new access token from Meta dashboard")
        
        elif e.response.status_code == 400:
            print("\n⚠️  Bad request. Possible reasons:")
            print("1. Business Account ID is incorrect")
            print("2. API version is outdated")
            print("\n💡 Solution: Verify your WHATSAPP_BUSINESS_ACCOUNT_ID")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
    
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    print("🚀 WhatsApp Phone Number ID Fetcher")
    print("=" * 70)
    print()
    get_whatsapp_phone_numbers()

