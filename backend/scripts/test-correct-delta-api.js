#!/usr/bin/env node
/**
 * Test Correct Delta Exchange API
 * Using the proper API format from Delta documentation
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
// Try the testnet API URL format
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🚀 DELTA EXCHANGE CORRECT API TEST');
console.log('===================================');
console.log(`API URL: ${BASE_URL}`);
console.log(`API Key: ${API_KEY}`);
console.log('');

function generateSignature(method, endpoint, timestamp, body = '') {
  const message = method + timestamp + endpoint + body;
  const signature = crypto
    .createHmac('sha256', API_SECRET)
    .update(message)
    .digest('hex');
  return signature;
}

async function testCorrectAPI() {
  console.log('📝 Testing authenticated endpoints...');
  
  // Test 1: Get wallet balance
  console.log('\n1. Testing wallet balance...');
  try {
    const endpoint = '/v2/wallet';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const signature = generateSignature('GET', endpoint, timestamp);
    
    const headers = {
      'api-key': API_KEY,
      'timestamp': timestamp,
      'signature': signature,
      'Content-Type': 'application/json'
    };

    console.log('Request headers:', headers);
    
    const response = await axios.get(`${BASE_URL}${endpoint}`, { headers });
    console.log('✓ Wallet data:', response.data);
  } catch (error) {
    console.log('✗ Wallet error:', error.response?.status, error.response?.data || error.message);
  }

  // Test 2: Get positions
  console.log('\n2. Testing positions...');
  try {
    const endpoint = '/v2/positions';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const signature = generateSignature('GET', endpoint, timestamp);
    
    const headers = {
      'api-key': API_KEY,
      'timestamp': timestamp,
      'signature': signature,
      'Content-Type': 'application/json'
    };
    
    const response = await axios.get(`${BASE_URL}${endpoint}`, { headers });
    console.log('✓ Positions:', response.data);
  } catch (error) {
    console.log('✗ Positions error:', error.response?.status, error.response?.data || error.message);
  }

  // Test 3: Try to place a small test order
  console.log('\n3. Testing order placement...');
  try {
    const endpoint = '/v2/orders';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    // Small test order for BTC-USDT
    const orderData = {
      product_id: 27,  // BTC-USDT perpetual
      size: 1,         // Minimal size
      side: 'buy',
      order_type: 'limit_order',
      limit_price: '90000',  // Well below market price for safety
      time_in_force: 'gtc'
    };
    
    const body = JSON.stringify(orderData);
    const signature = generateSignature('POST', endpoint, timestamp, body);
    
    const headers = {
      'api-key': API_KEY,
      'timestamp': timestamp,
      'signature': signature,
      'Content-Type': 'application/json'
    };
    
    console.log('Order payload:', orderData);
    
    const response = await axios.post(`${BASE_URL}${endpoint}`, orderData, { headers });
    console.log('✓ Order placed successfully:', response.data);
  } catch (error) {
    console.log('✗ Order error:', error.response?.status, error.response?.data || error.message);
  }
}

// Also test public endpoints
async function testPublicEndpoints() {
  console.log('\n📊 Testing public endpoints...');
  
  try {
    const response = await axios.get(`${BASE_URL}/v2/products`);
    console.log('✓ Public products endpoint works, found', response.data.result.length, 'products');
    
    // Find BTC-USDT product
    const btcProduct = response.data.result.find(p => p.symbol === 'BTCUSDT');
    if (btcProduct) {
      console.log('✓ Found BTCUSDT product:', btcProduct.id, btcProduct.description);
    }
  } catch (error) {
    console.log('✗ Public API error:', error.message);
  }
}

async function main() {
  await testPublicEndpoints();
  await testCorrectAPI();
}

main().catch(console.error);
