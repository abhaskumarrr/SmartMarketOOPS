#!/usr/bin/env node
/**
 * Delta Exchange Final Test
 * With CloudFront handling and proper headers
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://api.delta.exchange';

console.log('🚀 DELTA EXCHANGE FINAL TEST');
console.log('===========================');

// Configure axios defaults
axios.defaults.headers.common['User-Agent'] = 'delta-exchange-client/1.0.0';
axios.defaults.headers.common['Accept'] = 'application/json';
axios.defaults.headers.common['Accept-Encoding'] = 'gzip, deflate, br';

function generateSignature(method, endpoint, timestamp, body = '') {
    // Delta's v2 signature format - timestamp first
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
        const signature = generateSignature(method, endpoint, timestamp, body);

        const headers = {
            'api-key': API_KEY,
            'timestamp': timestamp,
            'signature': signature,
            'Content-Type': 'application/json',
            'User-Agent': 'delta-exchange-client/1.0.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate, br'
        };

        console.log('\nRequest Details:');
        console.log('URL:', `${BASE_URL}${endpoint}`);
        console.log('Method:', method);
        console.log('Headers:', headers);
        if (data) console.log('Data:', data);

        const response = await axios({
            method: method.toLowerCase(),
            url: `${BASE_URL}${endpoint}`,
            headers: headers,
            data: data,
            validateStatus: function (status) {
                return status < 500; // Accept all responses < 500
            }
        });

        if (response.status === 200) {
            console.log('\n✅ Success!');
            console.log('Response:', response.data);
        } else {
            console.log('\n⚠️ Non-200 Response:', response.status);
            console.log('Response:', response.data);
        }
        return response;
    } catch (error) {
        console.log('\n❌ Error:', error.response?.data || error.message);
        if (error.response?.data?.error?.code === 'invalid_api_key') {
            console.log('\n🔍 API Key Debug Info:');
            console.log('API Key:', API_KEY);
            console.log('API Key Length:', API_KEY.length);
            console.log('API Secret Length:', API_SECRET.length);
            console.log('Signature Length:', generateSignature('GET', '/v2/wallet', Math.floor(Date.now() / 1000).toString()).length);
        }
        return null;
    }
}

async function main() {
    // Test 1: Public endpoint
    console.log('\n1️⃣ Testing Public API...');
    try {
        const response = await axios.get(`${BASE_URL}/v2/products`);
        console.log('✅ Public API working');
        console.log(`Found ${response.data.result.length} products`);
    } catch (error) {
        console.log('❌ Public API error:', error.message);
        return;
    }

    // Test 2: Authentication test
    console.log('\n2️⃣ Testing Authentication...');
    const authResponse = await testEndpoint('GET', '/v2/wallet');
    
    if (authResponse?.data?.error?.code === 'invalid_api_key') {
        console.log('\n🔍 API Key Troubleshooting:');
        console.log('1. API Key Format:', API_KEY);
        console.log('2. API Key Length:', API_KEY.length);
        console.log('3. API Secret Length:', API_SECRET.length);
        console.log('4. Signature Format: timestamp + method + endpoint + body');
        return;
    }

    // Test 3: Place test order
    if (authResponse) {
        console.log('\n3️⃣ Testing Order Placement...');
        const orderData = {
            "product_id": 84,
            "size": 0.001,
            "side": "buy",
            "order_type": "limit_order",
            "limit_price": "85000",
            "time_in_force": "gtc"
        };
        await testEndpoint('POST', '/v2/orders', orderData);
    }
}

main().catch(console.error);
