#!/usr/bin/env python3
"""
Test Delta Exchange India Testnet Connection
"""

import requests
import hmac
import hashlib
import time
import json
from urllib.parse import urlencode

class DeltaExchangeTest:
    def __init__(self):
        self.api_key = "VuBmLRHofoTVFSAMvzOrjJKMU3x1Xt"
        self.api_secret = "YW6KCAIuoON1vBciRGzn5v0YYg7aKlzXOkYamZUMoUpknMT0PMh6ewVXd2DY"
        self.base_url = "https://testnet-api.delta.exchange"
        
    def generate_signature(self, method, endpoint, payload=""):
        """Generate signature for Delta Exchange API"""
        timestamp = str(int(time.time()))
        message = method + timestamp + endpoint + payload
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature, timestamp
    
    def make_request(self, method, endpoint, params=None, data=None):
        """Make authenticated request to Delta Exchange"""
        url = self.base_url + endpoint
        
        # Prepare payload
        if method == "GET" and params:
            query_string = urlencode(params)
            endpoint_with_params = endpoint + "?" + query_string
            payload = ""
        else:
            endpoint_with_params = endpoint
            payload = json.dumps(data) if data else ""
        
        # Generate signature
        signature, timestamp = self.generate_signature(method, endpoint_with_params, payload)
        
        # Headers
        headers = {
            'api-key': self.api_key,
            'signature': signature,
            'timestamp': timestamp,
            'Content-Type': 'application/json'
        }
        
        # Make request
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method == "POST":
                response = requests.post(url, data=payload, headers=headers)
            
            return response
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def test_connection(self):
        """Test basic connection to Delta Exchange"""
        print("🔗 Testing Delta Exchange India Testnet Connection...")
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🔑 API Key: {self.api_key[:10]}...")
        
        # Test 1: Get server time (public endpoint)
        print("\n📡 Test 1: Server Time (Public)")
        try:
            response = requests.get(f"{self.base_url}/v2/time")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server time: {data}")
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Get products (public endpoint)
        print("\n📡 Test 2: Available Products (Public)")
        try:
            response = requests.get(f"{self.base_url}/v2/products")
            if response.status_code == 200:
                data = response.json()
                if 'result' in data:
                    products = data['result'][:5]  # Show first 5
                    print(f"✅ Found {len(data['result'])} products. First 5:")
                    for product in products:
                        print(f"   - {product.get('symbol', 'N/A')}: {product.get('description', 'N/A')}")
                else:
                    print(f"✅ Response: {data}")
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Get account info (authenticated)
        print("\n🔐 Test 3: Account Info (Authenticated)")
        response = self.make_request("GET", "/v2/profile")
        if response and response.status_code == 200:
            data = response.json()
            print(f"✅ Account info retrieved successfully")
            if 'result' in data:
                profile = data['result']
                print(f"   - User ID: {profile.get('id', 'N/A')}")
                print(f"   - Email: {profile.get('email', 'N/A')}")
                print(f"   - KYC Status: {profile.get('kyc_status', 'N/A')}")
            else:
                print(f"   - Response: {data}")
        elif response:
            print(f"❌ Failed: {response.status_code} - {response.text}")
        else:
            print("❌ No response received")
        
        # Test 4: Get wallet balances (authenticated)
        print("\n💰 Test 4: Wallet Balances (Authenticated)")
        response = self.make_request("GET", "/v2/wallet/balances")
        if response and response.status_code == 200:
            data = response.json()
            print(f"✅ Wallet balances retrieved successfully")
            if 'result' in data:
                balances = data['result']
                print(f"   - Found {len(balances)} assets")
                for balance in balances[:5]:  # Show first 5
                    asset = balance.get('asset', {})
                    available = balance.get('available_balance', '0')
                    print(f"   - {asset.get('symbol', 'N/A')}: {available}")
            else:
                print(f"   - Response: {data}")
        elif response:
            print(f"❌ Failed: {response.status_code} - {response.text}")
        else:
            print("❌ No response received")
        
        # Test 5: Get positions (authenticated)
        print("\n📊 Test 5: Open Positions (Authenticated)")
        response = self.make_request("GET", "/v2/positions")
        if response and response.status_code == 200:
            data = response.json()
            print(f"✅ Positions retrieved successfully")
            if 'result' in data:
                positions = data['result']
                print(f"   - Found {len(positions)} positions")
                if positions:
                    for pos in positions[:3]:  # Show first 3
                        product = pos.get('product', {})
                        size = pos.get('size', '0')
                        print(f"   - {product.get('symbol', 'N/A')}: {size}")
                else:
                    print("   - No open positions")
            else:
                print(f"   - Response: {data}")
        elif response:
            print(f"❌ Failed: {response.status_code} - {response.text}")
        else:
            print("❌ No response received")
        
        print("\n" + "="*60)
        print("🎯 CONNECTION TEST SUMMARY")
        print("="*60)
        print("✅ If you see successful responses above, your API credentials are working!")
        print("🚀 Ready to deploy the SmartMarketOOPS system")
        print("⚠️  This is TESTNET - no real money will be used")
        print("💡 Perfect for testing your trading bot safely")

def main():
    """Run Delta Exchange connection test"""
    tester = DeltaExchangeTest()
    tester.test_connection()

if __name__ == "__main__":
    main()