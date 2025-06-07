#!/usr/bin/env node
/**
 * Test Delta Exchange Connection
 * Simple test to verify API credentials and connection
 */

require('dotenv').config();
const axios = require('axios');
const crypto = require('crypto');

// Test configuration
const config = {
  apiKey: process.env.DELTA_EXCHANGE_API_KEY || "AjTdJYCVE3aMZDAVQ2r6AQdmkU2mWc",
  apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || "R29RkXJfUIIt4o3vCDXImyg6q74JvByYltVKFH96UJG51lR1mm88PCGnMrUR",
  testnet: true
};

const baseUrl = config.testnet 
  ? 'https://cdn-ind.testnet.deltaex.org'
  : 'https://api.india.delta.exchange';

console.log('🧪 TESTING DELTA EXCHANGE CONNECTION');
console.log('═'.repeat(60));
console.log(`🔑 API Key: ${config.apiKey.substring(0, 8)}...${config.apiKey.substring(config.apiKey.length - 4)}`);
console.log(`🔒 API Secret: ${config.apiSecret.substring(0, 8)}...${config.apiSecret.substring(config.apiSecret.length - 4)}`);
console.log(`🌐 Base URL: ${baseUrl}`);
console.log(`🧪 Testnet: ${config.testnet}`);
console.log('═'.repeat(60));

// Generate signature for authenticated requests
function generateSignature(method, path, queryString, body, timestamp, apiSecret) {
  const message = method + timestamp + path + queryString + body;
  return crypto
    .createHmac('sha256', apiSecret)
    .update(message)
    .digest('hex');
}

// Test 1: Public API (no authentication required)
async function testPublicAPI() {
  try {
    console.log('\n📡 Test 1: Public API - Getting Products...');
    
    const response = await axios.get(`${baseUrl}/v2/products`);
    
    if (response.data.success) {
      const products = response.data.result;
      console.log(`✅ Public API working! Found ${products.length} products`);
      
      // Find BTC and ETH perpetuals
      const btcProduct = products.find(p => p.symbol === 'BTCUSD');
      const ethProduct = products.find(p => p.symbol === 'ETHUSD');
      
      if (btcProduct) {
        console.log(`🟡 BTC/USD Perpetual: ID ${btcProduct.id}, State: ${btcProduct.state}`);
      }
      if (ethProduct) {
        console.log(`🔵 ETH/USD Perpetual: ID ${ethProduct.id}, State: ${ethProduct.state}`);
      }
      
      return true;
    } else {
      console.log(`❌ Public API failed: ${response.data.error}`);
      return false;
    }
    
  } catch (error) {
    console.log(`❌ Public API error: ${error.message}`);
    if (error.response) {
      console.log(`   Status: ${error.response.status}`);
      console.log(`   Data:`, error.response.data);
    }
    return false;
  }
}

// Test 2: Authentication test
async function testAuthentication() {
  try {
    console.log('\n🔐 Test 2: Authentication - Getting Profile...');
    
    const method = 'GET';
    const path = '/v2/profile';
    const queryString = '';
    const body = '';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    const signature = generateSignature(method, path, queryString, body, timestamp, config.apiSecret);
    
    console.log(`📝 Signature message: "${method}${timestamp}${path}${queryString}${body}"`);
    console.log(`✍️ Generated signature: ${signature}`);
    
    const headers = {
      'api-key': config.apiKey,
      'signature': signature,
      'timestamp': timestamp,
      'Content-Type': 'application/json',
      'User-Agent': 'SmartMarketOOPS-Test-v1.0'
    };
    
    console.log(`📤 Request headers:`, headers);
    
    const response = await axios.get(`${baseUrl}${path}`, { headers });
    
    if (response.data.success) {
      console.log(`✅ Authentication successful!`);
      console.log(`👤 User ID: ${response.data.result.user_id}`);
      console.log(`📧 Email: ${response.data.result.email}`);
      return true;
    } else {
      console.log(`❌ Authentication failed: ${response.data.error}`);
      return false;
    }
    
  } catch (error) {
    console.log(`❌ Authentication error: ${error.message}`);
    if (error.response) {
      console.log(`   Status: ${error.response.status}`);
      console.log(`   Data:`, JSON.stringify(error.response.data, null, 2));
    }
    return false;
  }
}

// Test 3: Get balance
async function testBalance() {
  try {
    console.log('\n💰 Test 3: Getting Account Balance...');
    
    const method = 'GET';
    const path = '/v2/wallet/balances';
    const queryString = '';
    const body = '';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    const signature = generateSignature(method, path, queryString, body, timestamp, config.apiSecret);
    
    const headers = {
      'api-key': config.apiKey,
      'signature': signature,
      'timestamp': timestamp,
      'Content-Type': 'application/json',
      'User-Agent': 'SmartMarketOOPS-Test-v1.0'
    };
    
    const response = await axios.get(`${baseUrl}${path}`, { headers });
    
    if (response.data.success) {
      const balances = response.data.result;
      console.log(`✅ Balance retrieved successfully!`);
      
      balances.forEach(balance => {
        if (parseFloat(balance.balance) > 0) {
          console.log(`💼 ${balance.asset}: ${balance.balance} (Available: ${balance.available_balance})`);
        }
      });
      
      return true;
    } else {
      console.log(`❌ Balance retrieval failed: ${response.data.error}`);
      return false;
    }
    
  } catch (error) {
    console.log(`❌ Balance error: ${error.message}`);
    if (error.response) {
      console.log(`   Status: ${error.response.status}`);
      console.log(`   Data:`, JSON.stringify(error.response.data, null, 2));
    }
    return false;
  }
}

// Test 4: Get market data
async function testMarketData() {
  try {
    console.log('\n📈 Test 4: Getting Market Data...');
    
    const symbol = 'BTCUSD';
    const method = 'GET';
    const path = `/v2/tickers/${symbol}`;
    const queryString = '';
    const body = '';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    const signature = generateSignature(method, path, queryString, body, timestamp, config.apiSecret);
    
    const headers = {
      'api-key': config.apiKey,
      'signature': signature,
      'timestamp': timestamp,
      'Content-Type': 'application/json',
      'User-Agent': 'SmartMarketOOPS-Test-v1.0'
    };
    
    const response = await axios.get(`${baseUrl}${path}`, { headers });
    
    if (response.data.success) {
      const ticker = response.data.result;
      console.log(`✅ Market data retrieved successfully!`);
      console.log(`📊 ${symbol}: $${ticker.close || ticker.price} (24h: ${ticker.change_24h || 'N/A'})`);
      return true;
    } else {
      console.log(`❌ Market data failed: ${response.data.error}`);
      return false;
    }
    
  } catch (error) {
    console.log(`❌ Market data error: ${error.message}`);
    if (error.response) {
      console.log(`   Status: ${error.response.status}`);
      console.log(`   Data:`, JSON.stringify(error.response.data, null, 2));
    }
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('\n🚀 Starting Delta Exchange Connection Tests...\n');
  
  const results = {
    publicAPI: await testPublicAPI(),
    authentication: await testAuthentication(),
    balance: await testBalance(),
    marketData: await testMarketData()
  };
  
  console.log('\n📋 TEST RESULTS SUMMARY');
  console.log('═'.repeat(60));
  console.log(`📡 Public API: ${results.publicAPI ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`🔐 Authentication: ${results.authentication ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`💰 Balance: ${results.balance ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`📈 Market Data: ${results.marketData ? '✅ PASS' : '❌ FAIL'}`);
  console.log('═'.repeat(60));
  
  const passCount = Object.values(results).filter(Boolean).length;
  const totalTests = Object.keys(results).length;
  
  if (passCount === totalTests) {
    console.log('🎉 ALL TESTS PASSED! Delta Exchange connection is working perfectly!');
    console.log('🚀 Ready to run live trading system!');
  } else if (passCount >= 2) {
    console.log(`⚠️ ${passCount}/${totalTests} tests passed. Some issues detected.`);
    if (results.publicAPI && !results.authentication) {
      console.log('💡 Suggestion: Check API credentials - they might be incorrect or expired');
    }
  } else {
    console.log('❌ Multiple test failures. Please check your configuration.');
  }
  
  return passCount === totalTests;
}

// Execute tests
runAllTests().catch(error => {
  console.log(`❌ Test execution failed: ${error.message}`);
  process.exit(1);
});
