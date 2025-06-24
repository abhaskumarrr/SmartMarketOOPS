#!/usr/bin/env python3
"""
Simple Delta Exchange Test
"""

import requests
import json

def test_delta_simple():
    """Test Delta Exchange basic connectivity"""
    print("🔗 Testing Delta Exchange India Testnet...")
    
    base_url = "https://testnet-api.delta.exchange"
    
    # Test 1: Basic connectivity
    print("\n📡 Test 1: Basic Connectivity")
    try:
        response = requests.get(f"{base_url}/v2/products", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'result' in data:
                products = data['result']
                print(f"✅ Connected! Found {len(products)} products")
                
                # Look for BTC and ETH products
                btc_products = [p for p in products if 'BTC' in p.get('symbol', '') and 'USDT' in p.get('symbol', '')]
                eth_products = [p for p in products if 'ETH' in p.get('symbol', '') and 'USDT' in p.get('symbol', '')]
                
                print(f"📊 BTC products: {len(btc_products)}")
                print(f"📊 ETH products: {len(eth_products)}")
                
                if btc_products:
                    print(f"   Example BTC product: {btc_products[0].get('symbol')}")
                if eth_products:
                    print(f"   Example ETH product: {eth_products[0].get('symbol')}")
                
                return True
            else:
                print(f"❌ Unexpected response format: {data}")
                return False
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    success = test_delta_simple()
    
    print("\n" + "="*50)
    if success:
        print("✅ DELTA EXCHANGE CONNECTION SUCCESSFUL!")
        print("🚀 Ready to deploy SmartMarketOOPS system")
        print("⚠️  Using TESTNET (no real money)")
    else:
        print("❌ CONNECTION FAILED")
        print("💡 Check internet connection and try again")
    print("="*50)
    
    return success

if __name__ == "__main__":
    main()