#!/usr/bin/env node

/**
 * Test Order Placement and Cancellation
 * Focus on entry/exit functionality even with auth issues
 */

import DeltaExchangeAPI from '../services/deltaApiService';

async function testOrderPlacement() {
  console.log('🚀 TESTING ORDER PLACEMENT & CANCELLATION');
  console.log('=' .repeat(60));
  
  try {
    // Initialize Delta API for testnet
    const deltaApi = new DeltaExchangeAPI({ testnet: true });
    
    // Get credentials from environment
    const credentials = {
      key: process.env.DELTA_EXCHANGE_API_KEY || '',
      secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
    };

    if (!credentials.key || !credentials.secret) {
      console.log('❌ Delta API credentials not found');
      return;
    }

    console.log('✅ Credentials found');
    await deltaApi.initialize(credentials);
    console.log('✅ API client initialized');

    // Get markets first
    const markets = await deltaApi.getMarkets();
    console.log(`✅ Retrieved ${markets.length} markets`);
    
    // Find a suitable market for testing
    const btcMarket = markets.find(m => 
      m.symbol === 'BTC_USDT' || 
      (m.symbol.includes('BTC') && m.symbol.includes('USD') && m.is_active)
    );
    
    if (!btcMarket) {
      console.log('❌ No suitable BTC market found for testing');
      return;
    }
    
    console.log(`🎯 Using market: ${btcMarket.symbol}`);
    console.log(`   Type: ${btcMarket.contract_type}`);
    console.log(`   Active: ${btcMarket.is_active}`);
    
    // Test order placement
    console.log('\n📝 TESTING ORDER PLACEMENT...');

    let orderPlacementSuccess = false;
    let orderCancellationSuccess = false;

    try {
      // Place a very conservative limit order (won't execute)
      const orderParams = {
        symbol: btcMarket.symbol,
        side: 'buy' as const,
        size: 0.001, // Very small size
        type: 'limit' as const,
        price: 1000, // Very low price (won't execute)
        timeInForce: 'gtc' as const,
        clientOrderId: `test_${Date.now()}`,
      };
      
      console.log(`📊 Placing test order: ${orderParams.side} ${orderParams.size} ${orderParams.symbol} @ $${orderParams.price}`);
      
      const order = await deltaApi.placeOrder(orderParams);
      
      console.log('✅ ORDER PLACED SUCCESSFULLY!');
      console.log(`   Order ID: ${order.id}`);
      console.log(`   Status: ${order.status}`);
      console.log(`   Symbol: ${order.symbol}`);
      console.log(`   Side: ${order.side}`);
      console.log(`   Size: ${order.size}`);
      console.log(`   Price: $${order.price}`);

      orderPlacementSuccess = true;

      // Test order cancellation
      console.log('\n❌ TESTING ORDER CANCELLATION...');

      try {
        const cancelResult = await deltaApi.cancelOrder(order.id.toString());
        console.log('✅ ORDER CANCELLED SUCCESSFULLY!');
        console.log('   Cancel result:', cancelResult);

        orderCancellationSuccess = true;

        // Verify cancellation
        const activeOrders = await deltaApi.getActiveOrders();
        const stillActive = activeOrders.find(o => o.id === order.id);

        if (!stillActive) {
          console.log('✅ Order confirmed cancelled (not in active orders)');
        } else {
          console.log(`⚠️ Order still shows as: ${stillActive.status}`);
        }

      } catch (cancelError) {
        console.log('❌ Order cancellation failed:', cancelError.message);
      }
      
    } catch (orderError) {
      console.log('❌ Order placement failed:', orderError.message);
      
      // Check if it's an auth issue or market issue
      if (orderError.message.includes('expired_signature')) {
        console.log('🔧 Authentication issue - signature may need adjustment');
      } else if (orderError.message.includes('404')) {
        console.log('🔧 Endpoint issue - order endpoint may be different');
      } else {
        console.log('🔧 Other issue - check order parameters');
      }
    }
    
    console.log('\n🎉 ORDER TESTING COMPLETED');
    
    // Summary
    console.log('\n📊 SUMMARY:');
    console.log('✅ Market data access: WORKING');
    console.log('✅ API connection: WORKING');
    console.log('✅ Market selection: WORKING');

    if (orderPlacementSuccess) {
      console.log('✅ Order placement: WORKING');
    } else {
      console.log('⚠️ Order placement: NEEDS WORK');
    }

    if (orderCancellationSuccess) {
      console.log('✅ Order cancellation: WORKING');
    } else {
      console.log('⚠️ Order cancellation: NEEDS WORK');
    }
    
  } catch (error) {
    console.log('❌ Test failed:', error.message);
  }
}

testOrderPlacement().catch(console.error);
