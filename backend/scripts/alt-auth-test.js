#!/usr/bin/env node
/**
 * Alternative Delta Exchange Authentication Test
 * Using different signature format
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🔄 DELTA EXCHANGE ALTERNATIVE AUTH TEST');
console.log('=====================================');

async function testAuth() {
    const endpoint = '/v2/orders';
    const timestamp = Date.now().toString();
    
    // Try different signature format
    const message = `${timestamp}GET${endpoint}`;
    const signature = crypto
        .createHmac('sha256', API_SECRET)
        .update(message)
        .digest('hex');

    const headers = {
        'api-key': API_KEY,
        'timestamp': timestamp,
        'signature': signature,
        'Content-Type': 'application/json'
    };

    console.log('\nRequest Details:');
    console.log('Timestamp:', timestamp);
    console.log('Message:', message);
    console.log('Headers:', headers);

    try {
        const response = await axios.get(`${BASE_URL}${endpoint}`, { headers });
        console.log('\n✅ Success!');
        console.log(response.data);
    } catch (error) {
        console.log('\n❌ Error:', error.response?.data || error.message);
        
        if (error.response?.data?.error?.code === 'invalid_api_key') {
            console.log('\n⚠️ API Key Issue Detected');
            console.log('Please verify in Delta Exchange dashboard:');
            console.log('1. API key is activated');
            console.log('2. IP 49.37.8.175 is whitelisted');
            console.log('3. Testnet account is properly set up');
            console.log('\nIf issues persist, try:');
            console.log('1. Regenerating API keys');
            console.log('2. Creating a new testnet account');
            console.log('3. Contacting Delta Exchange support');
        }
    }
}

testAuth().catch(console.error);
