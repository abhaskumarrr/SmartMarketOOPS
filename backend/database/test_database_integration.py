#!/usr/bin/env python3
"""
Test script to verify database integration is working correctly
"""

import asyncio
import json
from database_service import DatabaseService

async def test_database_integration():
    """Test the database integration"""
    print("🧪 Testing Database Integration...")
    
    # Initialize database service
    db_service = DatabaseService()
    await db_service.initialize()
    
    try:
        # Test Redis operations
        print("\n📊 Testing Redis Operations...")
        
        # Get latest market data
        btc_data = await db_service.get_latest_market_data('BTCUSD')
        eth_data = await db_service.get_latest_market_data('ETHUSD')
        
        if btc_data:
            print(f"✅ BTCUSD Latest: ${btc_data['price']:.2f} (Volume: {btc_data.get('volume', 0):.2f})")
        else:
            print("⚠️ No BTCUSD data found in Redis")
            
        if eth_data:
            print(f"✅ ETHUSD Latest: ${eth_data['price']:.2f} (Volume: {eth_data.get('volume', 0):.2f})")
        else:
            print("⚠️ No ETHUSD data found in Redis")
        
        # Get recent trading signals
        signals = await db_service.get_recent_signals(5)
        print(f"\n🎯 Recent Trading Signals ({len(signals)} found):")
        
        for i, signal in enumerate(signals[:3], 1):
            symbol = signal.get('symbol', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            price = signal.get('price', 0)
            confidence = signal.get('confidence', 0)
            
            print(f"  {i}. {signal_type.upper()} {symbol} @ ${float(price):.2f} (Confidence: {float(confidence):.1%})")
        
        # Test QuestDB operations (if available)
        if db_service.questdb_pool:
            print("\n💾 Testing QuestDB Operations...")
            
            # Get historical data
            btc_history = await db_service.get_historical_data('BTCUSD', hours=1)
            print(f"✅ BTCUSD Historical Data: {len(btc_history)} records in last hour")
            
            if btc_history:
                latest = btc_history[0]
                print(f"   Latest: ${latest['price']:.2f} at {latest['timestamp']}")
        else:
            print("⚠️ QuestDB not available for historical data testing")
        
        print("\n✅ Database integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        
    finally:
        await db_service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_database_integration())
