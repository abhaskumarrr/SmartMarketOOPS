#!/usr/bin/env node
/**
 * Check IP and wait for API key activation
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🌐 DELTA EXCHANGE IP & ACTIVATION CHECK');
console.log('=======================================');

async function checkCurrentIP() {
  try {
    console.log('🔍 Checking your current IP address...');
    const response = await axios.get('https://api.ipify.org?format=json');
    console.log('✓ Your current IP:', response.data.ip);
    console.log('⚠️  Make sure this IP is whitelisted in your Delta Exchange API settings');
    return response.data.ip;
  } catch (error) {
    console.log('✗ Could not fetch IP:', error.message);
    return null;
  }
}

function generateSignature(method, endpoint, timestamp, body = '') {
  const message = method + timestamp + endpoint + body;
  const signature = crypto
    .createHmac('sha256', API_SECRET)
    .update(message)
    .digest('hex');
  return signature;
}

async function testAPIKey() {
  console.log('\n🔑 Testing API key authentication...');
  
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
    console.log('✓ API key working! Positions:', response.data);
    return true;
  } catch (error) {
    if (error.response?.status === 401) {
      const errorData = error.response.data;
      console.log('✗ Authentication failed:', errorData);
      
      if (errorData.error?.code === 'invalid_api_key') {
        console.log('💡 Possible causes:');
        console.log('   - API key needs more time to activate (wait 5 minutes)');
        console.log('   - IP address not whitelisted');
        console.log('   - API key copied incorrectly');
      }
    } else {
      console.log('✗ Request failed:', error.response?.status, error.response?.data || error.message);
    }
    return false;
  }
}

async function main() {
  await checkCurrentIP();
  
  console.log('\n⏱️  Testing API key activation...');
  console.log('   (The screenshot mentioned waiting 5 minutes for activation)');
  
  const maxRetries = 3;
  for (let i = 1; i <= maxRetries; i++) {
    console.log(`\n�� Attempt ${i}/${maxRetries}:`);
    
    const success = await testAPIKey();
    if (success) {
      console.log('\n🎉 API key is working! You can now place orders.');
      break;
    }
    
    if (i < maxRetries) {
      console.log('⏳ Waiting 30 seconds before retry...');
      await new Promise(resolve => setTimeout(resolve, 30000));
    } else {
      console.log('\n❌ API key still not working after all retries.');
      console.log('🔧 Please check:');
      console.log('   1. Wait the full 5 minutes for API key activation');
      console.log('   2. Verify your IP is whitelisted');
      console.log('   3. Double-check the API key and secret are correct');
    }
  }
}

main().catch(console.error);
