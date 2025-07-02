#!/usr/bin/env node
/**
 * Delta Exchange Test - Using Official Implementation
 * Based on Delta Exchange India documentation
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🔄 DELTA EXCHANGE RECOMMENDED IMPLEMENTATION TEST');
console.log('===============================================');

function getServerTime() {
    return Math.floor(Date.now() / 1000);
}

function generateSignature(timestamp, method, endpoint, body = '') {
    // Delta's specific signature format
    const signaturePayload = `${method}${endpoint}${timestamp}${body}`;
    console.log('Signature Payload:', signaturePayload);
    
    const signature = crypto
        .createHmac('sha256', API_SECRET)
        .update(signaturePayload)
        .digest('hex');
    
    return signature;
}

async function testEndpoint(method, endpoint, data = null) {
    try {
        const timestamp = getServerTime();
        const body = data ? JSON.stringify(data) : '';
        const signature = generateSignature(timestamp, method, endpoint, body);

        const headers = {
            'api-key': API_KEY,
            'timestamp': timestamp.toString(),
            'signature': signature,
            'Content-Type': 'application/json'
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
        if (error.response?.data?.error?.code === 'invalid_signature') {
            console.log('\nSignature Debug:');
            console.log('API Key:', API_KEY);
            console.log('Secret Length:', API_SECRET.length);
        }
        return false;
    }
}

async function main() {
    console.log('Testing with API Key:', API_KEY);
    
    // Test 1: Simple wallet endpoint
    console.log('\n1️⃣ Testing Wallet Access...');
    await testEndpoint('GET', '/v2/wallet');
    
    // Test 2: Positions endpoint
    console.log('\n2️⃣ Testing Positions...');
    await testEndpoint('GET', '/v2/positions');
    
    // Test 3: Place a test order
    console.log('\n3️⃣ Testing Order Placement...');
    const orderData = {
        "symbol": "BTCUSDT",
        "size": 0.001,
        "type": "limit",
        "side": "buy",
        "price": 85000,
        "time_in_force": "gtc"
    };
    await testEndpoint('POST', '/v2/orders', orderData);
}

main().catch(console.error);
