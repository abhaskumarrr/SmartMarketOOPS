#!/usr/bin/env node
/**
 * Debug Delta Exchange Authentication
 * Detailed debugging of authentication issues
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = process.env.DELTA_EXCHANGE_BASE_URL || 'https://cdn-ind.testnet.deltaex.org';

console.log('🔍 DELTA EXCHANGE AUTHENTICATION DEBUG');
console.log('=====================================');
console.log(`Base URL: ${BASE_URL}`);
console.log(`API Key: ${API_KEY}`);
console.log(`API Secret Length: ${API_SECRET ? API_SECRET.length : 0}`);
console.log('');

function generateAuthHeaders(method, path, body = '') {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const payload = method + timestamp + path + body;
  const signature = crypto
    .createHmac('sha256', API_SECRET)
    .update(payload)
    .digest('hex');

  return {
    'api-key': API_KEY,
    timestamp: timestamp,
    signature: signature,
    'Content-Type': 'application/json'
  };
}

async function testAuth() {
  console.log('🔑 Testing different authentication methods...');
  
  // Test 1: Basic wallet endpoint
  console.log('\n1. Testing /v2/wallet with standard auth...');
  try {
    const path = '/v2/wallet';
    const headers = generateAuthHeaders('GET', path);
    console.log('Headers:', JSON.stringify(headers, null, 2));
    
    const response = await axios.get(`${BASE_URL}${path}`, { headers });
    console.log('✓ Success:', response.data);
  } catch (error) {
    console.log('✗ Error:', error.response?.status, error.response?.data || error.message);
  }

  // Test 2: Try with different URL format
  console.log('\n2. Testing with api. subdomain...');
  try {
    const altUrl = 'https://api.testnet.delta.exchange';
    const path = '/v2/wallet';
    const headers = generateAuthHeaders('GET', path);
    
    const response = await axios.get(`${altUrl}${path}`, { headers });
    console.log('✓ Success with api subdomain:', response.data);
  } catch (error) {
    console.log('✗ Error with api subdomain:', error.response?.status, error.response?.data || error.message);
  }

  // Test 3: Test products endpoint (should work without auth)
  console.log('\n3. Testing public products endpoint...');
  try {
    const response = await axios.get(`${BASE_URL}/v2/products`);
    console.log('✓ Public API works, found', response.data.result.length, 'products');
  } catch (error) {
    console.log('✗ Public API error:', error.message);
  }

  // Test 4: Check timestamp format
  console.log('\n4. Verifying timestamp format...');
  const now = Math.floor(Date.now() / 1000);
  console.log('Current timestamp:', now);
  console.log('Timestamp length:', now.toString().length);
}

testAuth().catch(console.error);
