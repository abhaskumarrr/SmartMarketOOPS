#!/usr/bin/env node
/**
 * Quick Authentication Test
 * Simple script to quickly test if API key is working
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

function generateSignature(method, endpoint, timestamp, body = '') {
  const message = method + timestamp + endpoint + body;
  return crypto.createHmac('sha256', API_SECRET).update(message).digest('hex');
}

async function quickTest() {
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
    console.log('🎉 SUCCESS! API key is working.');
    console.log('   You can now run: node scripts/place-test-order.js');
    return true;
  } catch (error) {
    console.log('⏳ API key not yet active:', error.response?.data?.error?.code || 'Connection failed');
    return false;
  }
}

quickTest();
