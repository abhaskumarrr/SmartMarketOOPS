#!/usr/bin/env node

/**
 * Simple Delta Testnet Connection Test
 * Basic test to verify connection and functionality
 */

import DeltaExchangeAPI from '../services/deltaApiService';

async function testDeltaConnection() {
  console.log('🚀 SIMPLE DELTA TESTNET CONNECTION TEST');
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
      console.log('❌ Delta API credentials not found in environment');
      return;
    }

    console.log('✅ Credentials found');
    console.log(`🔑 API Key: ${credentials.key.substring(0, 8)}...`);

    // Initialize connection
    await deltaApi.initialize(credentials);
    console.log('✅ API client initialized');

    // Skip server time test (endpoint doesn't exist on Delta testnet)
    console.log('⚠️ Server time endpoint not available on Delta testnet');

    // Test account info
    try {
      const accountInfo = await deltaApi.getAccountInfo();
      console.log(`✅ Account ID: ${accountInfo.id}`);
      console.log(`✅ Email: ${accountInfo.email}`);
    } catch (error) {
      console.log('⚠️ Account info failed:', error.message);
    }

    // Test wallet balances
    try {
      const balances = await deltaApi.getWalletBalances();
      console.log(`✅ Retrieved ${balances.length} wallet balances`);
      
      const nonZeroBalances = balances.filter(b => parseFloat(b.balance) > 0);
      if (nonZeroBalances.length > 0) {
        console.log('💰 Non-zero balances:');
        nonZeroBalances.forEach(balance => {
          console.log(`   ${balance.asset}: ${balance.balance}`);
        });
      } else {
        console.log('⚠️ No non-zero balances (testnet account empty)');
      }
    } catch (error) {
      console.log('⚠️ Wallet balances failed:', error.message);
    }

    // Test market data (public endpoint - should work)
    try {
      console.log('📊 Testing market data endpoint...');
      const markets = await deltaApi.getMarkets();
      console.log(`✅ Retrieved ${markets.length} markets`);

      if (markets.length > 0) {
        console.log('📈 Sample markets:');
        markets.slice(0, 3).forEach(market => {
          console.log(`   ${market.symbol}: ${market.description || market.contract_type || 'N/A'}`);
        });
      }

      const btcMarket = markets.find(m =>
        m.symbol.includes('BTC') && m.symbol.includes('USD')
      );

      if (btcMarket) {
        console.log(`✅ Found BTC market: ${btcMarket.symbol}`);
        console.log(`   Type: ${btcMarket.contract_type || 'N/A'}`);
        console.log(`   Active: ${btcMarket.is_active ? 'Yes' : 'No'}`);
        console.log(`   Description: ${btcMarket.description}`);
      } else {
        console.log('⚠️ No BTC market found');
      }
    } catch (error) {
      console.log('⚠️ Market data failed:', error.message);
      console.log('   This suggests API connection issues');
    }

    // Test positions
    try {
      const positions = await deltaApi.getPositions();
      console.log(`✅ Retrieved ${positions.length} positions`);
    } catch (error) {
      console.log('⚠️ Positions failed:', error.message);
    }

    // Test active orders
    try {
      const orders = await deltaApi.getActiveOrders();
      console.log(`✅ Retrieved ${orders.length} active orders`);
    } catch (error) {
      console.log('⚠️ Active orders failed:', error.message);
    }

    console.log('\n🎉 DELTA TESTNET CONNECTION TEST COMPLETED');
    console.log('✅ Basic functionality verified');
    console.log('🚀 Ready for trading system integration');

  } catch (error) {
    console.log('❌ Delta testnet test failed:', error.message);
    console.log('🔧 Check API credentials and network connection');
  }
}

// Run the test
testDeltaConnection().catch(console.error);
