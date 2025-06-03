#!/usr/bin/env python3
"""
Test WebSocket Client for Real-time Market Data
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:3001"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
            # Subscribe to market data
            subscribe_message = {
                "type": "subscribe",
                "data": {
                    "channel": "market_data",
                    "symbols": ["BTCUSD", "ETHUSD"]
                }
            }
            
            await websocket.send(json.dumps(subscribe_message))
            print("📡 Subscribed to market data")
            
            # Listen for messages for 30 seconds
            timeout = 30
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'market_data':
                        market_data = data.get('data', {})
                        symbol = market_data.get('symbol', 'Unknown')
                        price = market_data.get('price', 0)
                        source = market_data.get('source', 'unknown')
                        print(f"📊 {symbol}: ${price:.2f} (source: {source})")
                    elif data.get('type') == 'trading_signal':
                        signal_data = data.get('data', {})
                        symbol = signal_data.get('symbol', 'Unknown')
                        signal_type = signal_data.get('signal_type', 'unknown')
                        confidence = signal_data.get('confidence', 0)
                        print(f"🎯 Signal: {signal_type.upper()} {symbol} (confidence: {confidence:.2%})")
                    elif data.get('type') == 'connection_status':
                        status = data.get('data', {}).get('status', 'unknown')
                        print(f"🔗 Connection status: {status}")
                    else:
                        print(f"📨 Received: {data.get('type', 'unknown')}")
                        
                except asyncio.TimeoutError:
                    print("⏰ No message received in 5 seconds, continuing...")
                    continue
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting WebSocket test client...")
    asyncio.run(test_websocket())
    print("🏁 Test completed")
