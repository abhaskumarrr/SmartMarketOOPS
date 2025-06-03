#!/usr/bin/env python3
"""
Investigation script to verify Delta Exchange data accuracy
"""

import asyncio
import aiohttp
import json
import requests
from datetime import datetime

async def investigate_delta_data():
    print("🔍 INVESTIGATING DELTA EXCHANGE DATA ACCURACY")
    print("=" * 60)
    
    # 1. Check what our backend is actually returning
    print("\n1. 📊 CHECKING OUR BACKEND DATA:")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:3002/api/market-data/ETHUSD") as response:
                if response.status == 200:
                    data = await response.json()
                    backend_data = data.get('data', {})
                    print(f"   Backend ETH Price: ${backend_data.get('price', 'N/A')}")
                    print(f"   Source: {data.get('source', 'unknown')}")
                    print(f"   Timestamp: {datetime.fromtimestamp(backend_data.get('timestamp', 0)/1000)}")
                else:
                    print(f"   ❌ Backend error: {response.status}")
    except Exception as e:
        print(f"   ❌ Backend connection error: {e}")
    
    # 2. Check if we're actually getting real data or mock data
    print("\n2. 🎭 CHECKING IF WE'RE USING MOCK DATA:")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:3002/api/market-data/status") as response:
                if response.status == 200:
                    data = await response.json()
                    status_data = data.get('data', {})
                    print(f"   Service Status: {status_data.get('status', 'unknown')}")
                    print(f"   Data Source: {status_data.get('source', 'unknown')}")
                    print(f"   Supported Symbols: {status_data.get('supportedSymbols', [])}")
                else:
                    print(f"   ❌ Status check error: {response.status}")
    except Exception as e:
        print(f"   ❌ Status check error: {e}")
    
    # 3. Try to get real market data from CoinGecko for comparison
    print("\n3. 🌐 GETTING REAL MARKET DATA FOR COMPARISON:")
    try:
        # CoinGecko API (free, no auth required)
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum,bitcoin&vs_currencies=usd", timeout=10)
        if response.status_code == 200:
            real_data = response.json()
            real_eth_price = real_data.get('ethereum', {}).get('usd', 'N/A')
            real_btc_price = real_data.get('bitcoin', {}).get('usd', 'N/A')
            print(f"   Real ETH Price (CoinGecko): ${real_eth_price}")
            print(f"   Real BTC Price (CoinGecko): ${real_btc_price}")
        else:
            print(f"   ❌ CoinGecko API error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ CoinGecko API error: {e}")
    
    # 4. Check Delta Exchange testnet directly
    print("\n4. 🏦 CHECKING DELTA EXCHANGE TESTNET DIRECTLY:")
    try:
        # Try Delta Exchange testnet API directly
        testnet_url = "https://cdn-ind.testnet.deltaex.org/products"
        response = requests.get(testnet_url, timeout=10)
        if response.status_code == 200:
            products = response.json()
            print(f"   ✅ Delta testnet accessible")
            print(f"   Available products: {len(products)} total")
            
            # Look for ETH-related products
            eth_products = [p for p in products if 'ETH' in p.get('symbol', '').upper()]
            print(f"   ETH-related products: {len(eth_products)}")
            
            for product in eth_products[:3]:  # Show first 3
                symbol = product.get('symbol', 'N/A')
                mark_price = product.get('mark_price', 'N/A')
                print(f"     - {symbol}: Mark Price = {mark_price}")
                
        else:
            print(f"   ❌ Delta testnet API error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Delta testnet API error: {e}")
    
    # 5. Check our CCXT symbol mapping
    print("\n5. 🗺️  CHECKING CCXT SYMBOL MAPPING:")
    try:
        import ccxt
        exchange = ccxt.deltaexchange({
            'sandbox': True,  # Testnet
            'enableRateLimit': True,
        })
        
        # Try to load markets without API keys (public data)
        markets = exchange.load_markets()
        print(f"   ✅ CCXT connected to Delta Exchange")
        print(f"   Total markets: {len(markets)}")
        
        # Check if ETH/USD exists
        eth_symbols = [symbol for symbol in markets.keys() if 'ETH' in symbol.upper()]
        print(f"   ETH-related symbols: {eth_symbols[:5]}")  # Show first 5
        
        # Check what our mapping resolves to
        symbol_mapping = {
            'ETHUSD': 'ETH/USD',
            'BTCUSD': 'BTC/USD'
        }
        
        for our_symbol, ccxt_symbol in symbol_mapping.items():
            if ccxt_symbol in markets:
                market_info = markets[ccxt_symbol]
                print(f"   ✅ {our_symbol} -> {ccxt_symbol}: {market_info.get('id', 'N/A')}")
            else:
                print(f"   ❌ {our_symbol} -> {ccxt_symbol}: NOT FOUND")
                
    except Exception as e:
        print(f"   ❌ CCXT investigation error: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 INVESTIGATION COMPLETE")

if __name__ == "__main__":
    asyncio.run(investigate_delta_data())
