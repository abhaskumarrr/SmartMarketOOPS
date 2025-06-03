#!/usr/bin/env node

/**
 * Debug Delta Exchange Endpoints
 * Test different endpoints to find working ones
 */

import DeltaExchangeAPI from '../services/deltaApiService';

async function debugEndpoints() {
  console.log('🔍 DEBUGGING DELTA EXCHANGE ENDPOINTS');
  console.log('=' .repeat(80));
  
  try {
    const deltaApi = new DeltaExchangeAPI({ testnet: true });
    
    const credentials = {
      key: process.env.DELTA_EXCHANGE_API_KEY || '',
      secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
    };

    if (!credentials.key || !credentials.secret) {
      console.log('❌ Credentials not found');
      return;
    }

    await deltaApi.initialize(credentials);
    console.log('✅ API initialized');

    // Test public endpoints first
    console.log('\n📊 PUBLIC ENDPOINTS:');
    try {
      const markets = await deltaApi.getMarkets();
      console.log(`✅ /v2/products: ${markets.length} markets`);
    } catch (error) {
      console.log(`❌ /v2/products: ${error.message}`);
    }

    // Test authenticated endpoints with different paths
    console.log('\n🔐 AUTHENTICATED ENDPOINTS:');
    
    const testEndpoints = [
      '/v2/user',
      '/v2/profile',
      '/v2/account',
      '/v2/wallet/balances',
      '/v2/orders',
      '/v2/positions',
      '/user',
      '/profile',
      '/account',
      '/wallet/balances',
      '/orders',
      '/positions'
    ];

    for (const endpoint of testEndpoints) {
      try {
        console.log(`Testing ${endpoint}...`);
        const response = await deltaApi['_makeRequest']({
          method: 'GET',
          endpoint: endpoint,
          authenticated: true
        });
        console.log(`✅ ${endpoint}: SUCCESS`);
        if (endpoint.includes('user') || endpoint.includes('profile') || endpoint.includes('account')) {
          console.log(`   Response keys: ${Object.keys(response).join(', ')}`);
        }
      } catch (error) {
        console.log(`❌ ${endpoint}: ${error.message}`);
      }
    }

    console.log('\n🎯 ENDPOINT TESTING COMPLETE');

  } catch (error) {
    console.log('❌ Debug failed:', error.message);
  }
}

debugEndpoints().catch(console.error);
