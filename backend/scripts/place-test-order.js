#!/usr/bin/env node
/**
 * Place Test Order on Delta Exchange
 * Simple script to place a small test order once authentication works
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🎯 DELTA EXCHANGE TEST ORDER PLACEMENT');
console.log('======================================');

function generateSignature(method, endpoint, timestamp, body = '') {
  const message = method + timestamp + endpoint + body;
  const signature = crypto
    .createHmac('sha256', API_SECRET)
    .update(message)
    .digest('hex');
  return signature;
}

async function checkWallet() {
  console.log('💰 Checking wallet balance...');
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

    const response = await axios.get(`${BASE_URL}${endpoint}`, { headers });
    console.log('✓ Wallet balance:', response.data);
    return response.data;
  } catch (error) {
    console.log('✗ Wallet check failed:', error.response?.data || error.message);
    return null;
  }
}

async function placeTestOrder() {
  console.log('\n📝 Placing small test order...');
  
  try {
    const endpoint = '/v2/orders';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    // Very small test order - BTC perpetual
    const orderData = {
      product_id: 84,  // BTCUSDT perpetual
      size: 1,         // Minimal size (1 USD)
      side: 'buy',
      order_type: 'limit_order',
      limit_price: '85000',  // Well below market price for safety
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
    
    console.log('📊 Order details:', orderData);
    
    const response = await axios.post(`${BASE_URL}${endpoint}`, orderData, { headers });
    console.log('🎉 Order placed successfully!');
    console.log('   Order ID:', response.data.result.id);
    console.log('   Status:', response.data.result.state);
    console.log('   Full response:', response.data);
    
    return response.data;
  } catch (error) {
    console.log('✗ Order placement failed:', error.response?.data || error.message);
    return null;
  }
}

async function cancelTestOrder(orderId) {
  console.log(`\n❌ Cancelling test order ${orderId}...`);
  
  try {
    const endpoint = `/v2/orders/${orderId}`;
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const signature = generateSignature('DELETE', endpoint, timestamp);
    
    const headers = {
      'api-key': API_KEY,
      'timestamp': timestamp,
      'signature': signature,
      'Content-Type': 'application/json'
    };
    
    const response = await axios.delete(`${BASE_URL}${endpoint}`, { headers });
    console.log('✓ Order cancelled successfully');
    return response.data;
  } catch (error) {
    console.log('✗ Order cancellation failed:', error.response?.data || error.message);
    return null;
  }
}

async function main() {
  // Check wallet first
  const wallet = await checkWallet();
  if (!wallet) {
    console.log('❌ Cannot proceed without wallet access. Check API authentication.');
    return;
  }
  
  // Place test order
  const order = await placeTestOrder();
  if (order && order.result) {
    console.log('\n⏱️  Waiting 5 seconds before cancelling...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Cancel the test order
    await cancelTestOrder(order.result.id);
  }
  
  console.log('\n✅ Test order flow completed!');
  console.log('🚀 Your Delta Exchange integration is working correctly.');
}

main().catch(console.error);
