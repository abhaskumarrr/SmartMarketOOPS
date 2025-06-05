// Test the service directly with the same approach as our working test
require('dotenv').config();

const crypto = require('crypto');
const axios = require('axios');

// Use the exact same credentials and approach as our working test
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://cdn-ind.testnet.deltaex.org';

console.log('🔍 Testing service approach with working credentials...');
console.log('API Key:', API_KEY);
console.log('API Secret:', API_SECRET ? '***' + API_SECRET.slice(-4) : 'undefined');

function generateSignature(secret, message) {
    return crypto.createHmac('sha256', secret).update(message).digest('hex');
}

async function testServiceApproach() {
    try {
        const method = 'GET';
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const path = '/v2/wallet/balances';
        const queryString = '';
        const body = '';
        
        const message = method + timestamp + path + queryString + body;
        const signature = generateSignature(API_SECRET, message);
        
        console.log('\n🔐 Signature generation:');
        console.log('Method:', method);
        console.log('Timestamp:', timestamp);
        console.log('Path:', path);
        console.log('Message:', message);
        console.log('Signature:', signature);
        
        const headers = {
            'api-key': API_KEY,
            'timestamp': timestamp,
            'signature': signature,
            'User-Agent': 'SmartMarketOOPS-v1.0',
            'Content-Type': 'application/json'
        };
        
        console.log('\n📡 Making request...');
        const response = await axios.get(`${BASE_URL}${path}`, { headers });
        
        console.log('✅ Success!');
        console.log('Response:', JSON.stringify(response.data, null, 2));
        
    } catch (error) {
        console.error('❌ Error:', error.response?.data || error.message);
    }
}

testServiceApproach();
