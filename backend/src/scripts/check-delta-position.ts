#!/usr/bin/env node

/**
 * Check Current Delta Exchange Position
 * Clean script to view your current position
 */

import DeltaExchangeAPI from '../services/deltaApiService';
import { logger } from '../utils/logger';

async function checkDeltaPosition() {
  console.log('🔍 CHECKING YOUR DELTA EXCHANGE POSITION');
  console.log('=' .repeat(80));
  
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
      console.log('🔧 Please set DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET');
      return;
    }

    console.log('✅ Credentials found');
    console.log(`🔑 API Key: ${credentials.key.substring(0, 8)}...`);

    // Initialize connection
    await deltaApi.initialize(credentials);
    console.log('✅ API client initialized');

    // Test 1: Get account info
    console.log('\n👤 ACCOUNT INFORMATION:');
    try {
      const accountInfo = await deltaApi.getAccountInfo();
      console.log('✅ Account access successful');
      console.log(`   Account ID: ${accountInfo.id}`);
      console.log(`   Email: ${accountInfo.email}`);
      console.log(`   Name: ${accountInfo.name}`);
      console.log(`   Verified: ${accountInfo.is_verified}`);
    } catch (error) {
      console.log('❌ Account info failed:', error.message);
      if (error.message.includes('ip_not_whitelisted')) {
        console.log('🔧 IP needs to be whitelisted in Delta Exchange account');
        console.log('🌐 Current IP needs whitelisting for API access');
        return;
      }
    }

    // Test 2: Get wallet balances
    console.log('\n💰 WALLET BALANCES:');
    try {
      const balances = await deltaApi.getWalletBalances();
      console.log(`✅ Retrieved ${balances.length} wallet balances`);
      
      const nonZeroBalances = balances.filter(b => parseFloat(b.balance) > 0);
      if (nonZeroBalances.length > 0) {
        console.log('💰 Your balances:');
        nonZeroBalances.forEach(balance => {
          console.log(`   ${balance.asset}: ${balance.balance} (Available: ${balance.available_balance})`);
        });
      } else {
        console.log('⚠️ No balances found (testnet account may be empty)');
      }
    } catch (error) {
      console.log('❌ Wallet balances failed:', error.message);
    }

    // Test 3: Get current positions
    console.log('\n📈 YOUR CURRENT POSITIONS:');
    try {
      const positions = await deltaApi.getPositions();
      console.log(`✅ Retrieved ${positions.length} positions`);
      
      if (positions.length > 0) {
        console.log('🔥 ACTIVE POSITIONS:');
        positions.forEach((position, index) => {
          console.log(`\n   Position ${index + 1}:`);
          console.log(`     Symbol: ${position.symbol}`);
          console.log(`     Size: ${position.size}`);
          console.log(`     Entry Price: $${position.entry_price}`);
          console.log(`     Current PnL: $${position.realized_pnl || 0}`);
          console.log(`     Unrealized PnL: $${position.unrealized_pnl || 0}`);
          console.log(`     Side: ${parseFloat(position.size) > 0 ? 'LONG' : 'SHORT'}`);
        });
      } else {
        console.log('✅ No active positions found');
      }
    } catch (error) {
      console.log('❌ Positions failed:', error.message);
    }

    // Test 4: Get active orders
    console.log('\n📝 YOUR ACTIVE ORDERS:');
    try {
      const orders = await deltaApi.getActiveOrders();
      console.log(`✅ Retrieved ${orders.length} active orders`);
      
      if (orders.length > 0) {
        console.log('📋 ACTIVE ORDERS:');
        orders.forEach((order, index) => {
          console.log(`\n   Order ${index + 1}:`);
          console.log(`     Symbol: ${order.symbol}`);
          console.log(`     Side: ${order.side}`);
          console.log(`     Size: ${order.size}`);
          console.log(`     Price: $${order.limit_price || order.price}`);
          console.log(`     Status: ${order.status}`);
          console.log(`     Order ID: ${order.id}`);
        });
      } else {
        console.log('✅ No active orders found');
      }
    } catch (error) {
      console.log('❌ Active orders failed:', error.message);
    }

    // Test 5: Get markets for context
    console.log('\n📊 AVAILABLE MARKETS:');
    try {
      const markets = await deltaApi.getMarkets();
      const btcMarkets = markets.filter(m => m.symbol.includes('BTC'));
      const ethMarkets = markets.filter(m => m.symbol.includes('ETH'));
      const solMarkets = markets.filter(m => m.symbol.includes('SOL'));
      
      console.log(`✅ Total markets: ${markets.length}`);
      console.log(`   BTC markets: ${btcMarkets.length}`);
      console.log(`   ETH markets: ${ethMarkets.length}`);
      console.log(`   SOL markets: ${solMarkets.length}`);
      
      if (btcMarkets.length > 0) {
        console.log(`   Main BTC market: ${btcMarkets[0].symbol}`);
      }
    } catch (error) {
      console.log('❌ Markets failed:', error.message);
    }

    console.log('\n🎯 POSITION ANALYSIS COMPLETE');
    console.log('=' .repeat(80));

  } catch (error) {
    console.log('❌ Position check failed:', error.message);
    console.log('🔧 Check API credentials and network connection');
  }
}

checkDeltaPosition().catch(console.error);
