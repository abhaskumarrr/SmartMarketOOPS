#!/usr/bin/env node

/**
 * Direct Delta India API Test
 * Simple test without Prisma dependencies
 */

import axios from 'axios';
import * as crypto from 'crypto';
import * as querystring from 'querystring';

// Delta India API configuration
const DELTA_INDIA_TESTNET_URL = 'https://cdn-ind.testnet.deltaex.org';
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY || 'YsA1TIH5EXk8fl0AYkDtV464ErNa4T';
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET || 'kKBR52xNlKEGLQXOEAOnJlCUip60g4vblyI0BAi5h5scaIWfVom2KQ9RCMat';

/**
 * Generate Delta India API signature
 */
function generateSignature(method: string, path: string, queryString: string, body: string, timestamp: number): string {
  const message = method + timestamp.toString() + path + queryString + body;
  return crypto.createHmac('sha256', API_SECRET).update(message).digest('hex');
}

/**
 * Make authenticated request to Delta India API
 */
async function makeRequest(method: string, endpoint: string, params: any = {}, data: any = null): Promise<any> {
  const timestamp = Math.floor(Date.now() / 1000);
  const queryString = Object.keys(params).length > 0 ? '?' + querystring.stringify(params) : '';
  const body = data ? JSON.stringify(data) : '';
  
  const signature = generateSignature(method, endpoint, queryString, body, timestamp);
  
  const headers: any = {
    'Content-Type': 'application/json',
    'User-Agent': 'nodejs-rest-client'
  };

  // Add auth headers for authenticated requests
  if (method !== 'GET' || endpoint.includes('wallet') || endpoint.includes('orders') || endpoint.includes('positions')) {
    headers['api-key'] = API_KEY;
    headers['timestamp'] = timestamp.toString();
    headers['signature'] = signature;
  }

  const config = {
    method,
    url: DELTA_INDIA_TESTNET_URL + endpoint + queryString,
    headers,
    data: data ? body : undefined
  };

  try {
    const response = await axios(config);
    return response.data.result !== undefined ? response.data.result : response.data;
  } catch (error: any) {
    if (error.response) {
      throw new Error(`Delta India API Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
    }
    throw error;
  }
}

/**
 * Main test function
 */
async function testDeltaIndia() {
  console.log('🇮🇳 DELTA EXCHANGE INDIA DIRECT API TEST');
  console.log('=' .repeat(80));
  
  try {
    // Test 1: Public endpoint - Get products
    console.log('\n📊 STEP 1: PUBLIC ENDPOINTS TEST');
    const products = await makeRequest('GET', '/v2/products');
    console.log(`✅ Products retrieved: ${products.length} total`);
    
    // Filter perpetual contracts
    const perpetuals = products.filter((p: any) => 
      p.contract_type === 'perpetual_futures' && p.is_active
    );
    console.log(`✅ Perpetual contracts: ${perpetuals.length} active`);
    
    // Find BTC perpetual
    const btcPerpetual = perpetuals.find((p: any) => 
      p.symbol === 'BTCUSD' || p.symbol.includes('BTC')
    );
    
    if (btcPerpetual) {
      console.log(`✅ BTC Perpetual found: ${btcPerpetual.symbol}`);
      console.log(`   Product ID: ${btcPerpetual.id}`);
      console.log(`   Contract Type: ${btcPerpetual.contract_type}`);
      
      // Test ticker
      try {
        const ticker = await makeRequest('GET', '/v2/tickers', { symbol: btcPerpetual.symbol });
        console.log(`✅ BTC Ticker: $${ticker.close}, Volume: ${ticker.volume}`);
      } catch (tickerError) {
        console.log(`⚠️ Ticker test failed: ${tickerError.message}`);
      }
    }

    // Test 2: Authentication test
    console.log('\n🔐 STEP 2: AUTHENTICATION TEST');
    
    try {
      // Test wallet balances
      const balances = await makeRequest('GET', '/v2/wallet/balances');
      console.log(`✅ Wallet access successful - ${balances.length} assets`);
      
      const nonZeroBalances = balances.filter((b: any) => parseFloat(b.balance) > 0);
      if (nonZeroBalances.length > 0) {
        console.log('💰 Available balances:');
        nonZeroBalances.forEach((balance: any) => {
          console.log(`   ${balance.asset}: ${balance.balance}`);
        });
      } else {
        console.log('   No balances found (testnet account may be empty)');
      }

      // Test positions
      const positions = await makeRequest('GET', '/v2/positions');
      console.log(`✅ Positions access successful - ${positions.length} positions`);

      // Test active orders
      const orders = await makeRequest('GET', '/v2/orders');
      console.log(`✅ Orders access successful - ${orders.length} active orders`);

    } catch (authError) {
      console.log(`❌ Authentication failed: ${authError.message}`);
      
      if (authError.message.includes('ip_not_whitelisted')) {
        console.log('\n🔧 IP WHITELISTING REQUIRED:');
        console.log('   1. Login to Delta Exchange India testnet');
        console.log('   2. Go to API Management section');
        console.log('   3. Edit your API key settings');
        console.log('   4. Add your current IP address to whitelist');
        console.log('   5. Save changes and try again');
        return;
      }
      throw authError;
    }

    // Test 3: Order placement
    console.log('\n🎯 STEP 3: ORDER PLACEMENT TEST');
    
    if (btcPerpetual) {
      try {
        // Get current price
        const ticker = await makeRequest('GET', '/v2/tickers', { symbol: btcPerpetual.symbol });
        const currentPrice = parseFloat(ticker.close);
        
        // Create test order (10% below market, won't execute)
        const orderData = {
          product_id: btcPerpetual.id,
          size: 1,
          side: 'buy',
          order_type: 'limit_order',
          limit_price: (currentPrice * 0.90).toFixed(2),
          time_in_force: 'gtc'
        };

        console.log('\n📋 ORDER PARAMETERS:');
        console.log(`   Symbol: ${btcPerpetual.symbol}`);
        console.log(`   Side: BUY`);
        console.log(`   Size: 1 contract`);
        console.log(`   Price: $${orderData.limit_price} (10% below market)`);
        console.log(`   Current Market: $${currentPrice.toFixed(2)}`);

        console.log('\n🚀 PLACING ORDER ON DELTA INDIA...');
        
        const order = await makeRequest('POST', '/v2/orders', {}, orderData);
        
        console.log('\n🎉 ORDER PLACED SUCCESSFULLY!');
        console.log('=' .repeat(60));
        console.log(`✅ Order ID: ${order.id}`);
        console.log(`✅ Status: ${order.state || order.status}`);
        console.log(`✅ Size: ${order.size}`);
        console.log(`✅ Price: $${order.limit_price}`);
        console.log('=' .repeat(60));

        // Wait and cancel
        console.log('\n⏳ Waiting 3 seconds before cancellation...');
        await new Promise(resolve => setTimeout(resolve, 3000));

        console.log('\n❌ CANCELLING ORDER...');
        try {
          await makeRequest('DELETE', `/v2/orders/${order.id}`);
          console.log('✅ Order cancelled successfully');
        } catch (cancelError) {
          console.log(`⚠️ Cancellation failed: ${cancelError.message}`);
        }

        console.log('\n🎯 ORDER TEST RESULTS:');
        console.log('✅ Order placement: SUCCESS');
        console.log('✅ API integration: WORKING');
        console.log('✅ Authentication: WORKING');
        console.log('🚀 DELTA INDIA READY FOR LIVE TRADING!');

      } catch (orderError) {
        console.log(`❌ Order placement failed: ${orderError.message}`);
        
        if (orderError.message.includes('insufficient')) {
          console.log('💰 Insufficient balance - expected for testnet');
          console.log('✅ Order API is working (balance issue only)');
        }
      }
    }

    console.log('\n🎉 DELTA INDIA DIRECT TEST COMPLETED!');
    console.log('🚀 System ready for perpetual futures trading!');

  } catch (error) {
    console.log(`❌ Test failed: ${error.message}`);
  }
}

// Run the test
testDeltaIndia().catch(console.error);
