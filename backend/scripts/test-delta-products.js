#!/usr/bin/env node
/**
 * Test script to fetch Delta Exchange India testnet products and verify API connectivity
 */

const axios = require('axios');
const crypto = require('crypto');

// Delta Exchange India testnet configuration
const BASE_URL = 'https://cdn-ind.testnet.deltaex.org';
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY || 'uS2N0I4V37gMNJgbTjX8a33WPWv3GK';
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET || 'hJwxEd1wCpMTYg5iSQKDnreX9IVlc4mcYegR5ojJzvQ5UVOiUhP7cF9u21To';

/**
 * Generate signature for Delta Exchange API
 */
function generateSignature(method, path, queryString, body, timestamp) {
  const message = method + timestamp + path + queryString + body;
  return crypto
    .createHmac('sha256', API_SECRET)
    .update(message)
    .digest('hex');
}

/**
 * Make authenticated request to Delta Exchange API
 */
async function makeAuthenticatedRequest(method, path, params = {}, data = null) {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const queryString = Object.keys(params).length > 0 ? '?' + new URLSearchParams(params).toString() : '';
  const body = data ? JSON.stringify(data) : '';
  
  const signature = generateSignature(method, path, queryString, body, timestamp);
  
  const headers = {
    'api-key': API_KEY,
    'signature': signature,
    'timestamp': timestamp,
    'Content-Type': 'application/json',
    'User-Agent': 'SmartMarketOOPS-TestScript-v1.0'
  };

  console.log(`🔍 Making request: ${method} ${path}${queryString}`);
  console.log(`📝 Signature message: "${method}${timestamp}${path}${queryString}${body}"`);
  console.log(`✍️ Generated signature: ${signature}`);

  try {
    const response = await axios({
      method,
      url: BASE_URL + path + queryString,
      data: data || undefined,
      headers
    });

    return response.data;
  } catch (error) {
    console.error(`❌ API Error: ${error.message}`);
    if (error.response) {
      console.error(`Response status: ${error.response.status}`);
      console.error(`Response data:`, JSON.stringify(error.response.data, null, 2));
    }
    throw error;
  }
}

/**
 * Test public endpoints (no authentication required)
 */
async function testPublicEndpoints() {
  console.log('\n📊 TESTING PUBLIC ENDPOINTS:');
  
  try {
    // Test products endpoint
    console.log('\n1. Testing /v2/products endpoint...');
    const response = await axios.get(`${BASE_URL}/v2/products`);
    
    if (response.data.success) {
      const products = response.data.result;
      console.log(`✅ Found ${products.length} products`);
      
      // Find BTC and ETH perpetual futures
      const btcProducts = products.filter(p => 
        p.symbol.includes('BTC') && 
        p.contract_type === 'perpetual_futures' &&
        p.state === 'live'
      );
      
      const ethProducts = products.filter(p => 
        p.symbol.includes('ETH') && 
        p.contract_type === 'perpetual_futures' &&
        p.state === 'live'
      );
      
      console.log('\n🪙 BTC Perpetual Futures:');
      btcProducts.forEach(p => {
        console.log(`  - ${p.symbol} (ID: ${p.id}) - ${p.description}`);
      });
      
      console.log('\n🪙 ETH Perpetual Futures:');
      ethProducts.forEach(p => {
        console.log(`  - ${p.symbol} (ID: ${p.id}) - ${p.description}`);
      });
      
      return { btcProducts, ethProducts };
    } else {
      console.error('❌ Products request failed:', response.data.error);
      return null;
    }
  } catch (error) {
    console.error('❌ Public endpoint test failed:', error.message);
    return null;
  }
}

/**
 * Test authenticated endpoints
 */
async function testAuthenticatedEndpoints() {
  console.log('\n🔐 TESTING AUTHENTICATED ENDPOINTS:');
  
  try {
    // Test wallet balance
    console.log('\n1. Testing /v2/wallet/balances endpoint...');
    const balanceResponse = await makeAuthenticatedRequest('GET', '/v2/wallet/balances');
    
    if (balanceResponse.success) {
      console.log('✅ Balance request successful');
      console.log('💰 Balances:', balanceResponse.result);
    } else {
      console.error('❌ Balance request failed:', balanceResponse.error);
    }
    
    // Test positions with required parameter
    console.log('\n2. Testing /v2/positions endpoint with BTC...');
    const positionsResponse = await makeAuthenticatedRequest('GET', '/v2/positions', {
      underlying_asset_symbol: 'BTC'
    });

    if (positionsResponse.success) {
      console.log('✅ Positions request successful');
      console.log('📊 BTC Positions:', positionsResponse.result);
    } else {
      console.error('❌ Positions request failed:', positionsResponse.error);
    }

    // Test positions with ETH
    console.log('\n3. Testing /v2/positions endpoint with ETH...');
    const ethPositionsResponse = await makeAuthenticatedRequest('GET', '/v2/positions', {
      underlying_asset_symbol: 'ETH'
    });

    if (ethPositionsResponse.success) {
      console.log('✅ ETH Positions request successful');
      console.log('📊 ETH Positions:', ethPositionsResponse.result);
    } else {
      console.error('❌ ETH Positions request failed:', ethPositionsResponse.error);
    }
    
    return true;
  } catch (error) {
    console.error('❌ Authenticated endpoint test failed:', error.message);
    return false;
  }
}

/**
 * Test ticker endpoints
 */
async function testTickerEndpoints(products) {
  console.log('\n📈 TESTING TICKER ENDPOINTS:');
  
  if (!products || !products.btcProducts.length) {
    console.log('⚠️ No BTC products found, skipping ticker tests');
    return;
  }
  
  const btcSymbol = products.btcProducts[0].symbol;
  
  try {
    console.log(`\n1. Testing ticker for ${btcSymbol}...`);
    const tickerResponse = await axios.get(`${BASE_URL}/v2/tickers/${btcSymbol}`);
    
    if (tickerResponse.data.success) {
      console.log('✅ Ticker request successful');
      console.log('📊 Ticker data:', tickerResponse.data.result);
    } else {
      console.error('❌ Ticker request failed:', tickerResponse.data.error);
    }
  } catch (error) {
    console.error('❌ Ticker endpoint test failed:', error.message);
  }
}

/**
 * Main test function
 */
async function main() {
  console.log('🚀 Delta Exchange India Testnet API Test');
  console.log('==========================================');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`API Key: ${API_KEY ? API_KEY.substring(0, 8) + '...' : 'NOT SET'}`);
  console.log(`API Secret: ${API_SECRET ? API_SECRET.substring(0, 8) + '...' : 'NOT SET'}`);
  
  if (!API_KEY || !API_SECRET) {
    console.error('❌ API credentials not found in environment variables');
    process.exit(1);
  }
  
  try {
    // Test public endpoints first
    const products = await testPublicEndpoints();
    
    // Test ticker endpoints
    await testTickerEndpoints(products);
    
    // Test authenticated endpoints
    await testAuthenticatedEndpoints();
    
    console.log('\n✅ All tests completed!');
    
  } catch (error) {
    console.error('\n❌ Test suite failed:', error.message);
    process.exit(1);
  }
}

// Run the test
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { testPublicEndpoints, testAuthenticatedEndpoints, testTickerEndpoints };
