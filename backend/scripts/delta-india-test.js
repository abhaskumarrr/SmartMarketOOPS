#!/usr/bin/env node
/**
 * Delta Exchange India Test
 * Using India-specific endpoints
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://api.delta.exchange'; // Using main API endpoint

console.log('🇮🇳 DELTA EXCHANGE INDIA TEST');
console.log('============================');

function generateSignature(timestamp, method, endpoint, body = '') {
    const signaturePayload = `${timestamp}${method}${endpoint}${body}`;
    console.log('Signature Payload:', signaturePayload);
    
    return crypto
        .createHmac('sha256', API_SECRET)
        .update(signaturePayload)
        .digest('hex');
}

async function testEndpoint(method, endpoint, data = null) {
    try {
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const body = data ? JSON.stringify(data) : '';
        const signature = generateSignature(timestamp, method, endpoint, body);

        const headers = {
            'api-key': API_KEY,
            'timestamp': timestamp,
            'signature': signature,
            'Content-Type': 'application/json',
            'User-Agent': 'Delta-Exchange-Client'
        };

        console.log('\nRequest Details:');
        console.log('URL:', `${BASE_URL}${endpoint}`);
        console.log('Method:', method);
        console.log('Timestamp:', timestamp);
        console.log('Headers:', headers);
        if (data) console.log('Data:', data);

        const response = await axios({
            method: method.toLowerCase(),
            url: `${BASE_URL}${endpoint}`,
            headers: headers,
            data: data
        });

        console.log('\n✅ Success!');
        console.log('Response:', response.data);
        return true;
    } catch (error) {
        console.log('\n❌ Error:', error.response?.data || error.message);
        return false;
    }
}

async function main() {
    // Test public endpoint first
    console.log('\n1️⃣ Testing Public Endpoint...');
    try {
        const response = await axios.get(`${BASE_URL}/v2/products`);
        console.log('✅ Public API working');
        console.log(`Found ${response.data.result.length} products`);
    } catch (error) {
        console.log('❌ Public API error:', error.message);
        return;
    }

    // Test authenticated endpoints
    console.log('\n2️⃣ Testing Wallet...');
    await testEndpoint('GET', '/v2/wallet');
    
    console.log('\n3️⃣ Testing Orders...');
    const orderData = {
        "product_id": 84,  // BTCUSDT
        "size": 0.001,
        "side": "buy",
        "order_type": "limit_order",
        "limit_price": "85000",
        "time_in_force": "gtc"
    };
    await testEndpoint('POST', '/v2/orders', orderData);
}

main().catch(console.error);
