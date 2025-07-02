#!/usr/bin/env node
/**
 * Delta Exchange API Diagnostic Test
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🔍 DELTA EXCHANGE API DIAGNOSTIC');
console.log('===============================');

// Test public endpoints first
async function testPublicEndpoints() {
    console.log('\n1️⃣ Testing Public Endpoints...');
    try {
        const response = await axios.get(`${BASE_URL}/v2/products`);
        console.log('✅ Public API working');
        console.log(`   Found ${response.data.result.length} products`);
        return true;
    } catch (error) {
        console.log('❌ Public API error:', error.message);
        return false;
    }
}

// Test authentication with different endpoints
async function testAuthEndpoints() {
    console.log('\n2️⃣ Testing Authentication...');
    
    const endpoints = [
        { path: '/v2/wallet', method: 'GET' },
        { path: '/v2/positions', method: 'GET' },
        { path: '/v2/orders', method: 'GET' }
    ];

    for (const endpoint of endpoints) {
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const message = endpoint.method + timestamp + endpoint.path;
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

        try {
            console.log(`\nTesting ${endpoint.path}...`);
            console.log('Request Headers:', headers);
            
            const response = await axios({
                method: endpoint.method,
                url: `${BASE_URL}${endpoint.path}`,
                headers: headers
            });
            
            console.log('✅ Success!');
            console.log('Response:', response.data);
            return true;
        } catch (error) {
            console.log('❌ Error:', error.response?.data || error.message);
            if (error.response?.data?.error?.code === 'invalid_api_key') {
                console.log('\n🔍 Diagnostic Information:');
                console.log('1. API Key Length:', API_KEY.length);
                console.log('2. API Secret Length:', API_SECRET.length);
                console.log('3. Timestamp:', timestamp);
                console.log('4. Signature Length:', signature.length);
            }
        }
    }
    return false;
}

// Check IP address
async function checkIP() {
    console.log('\n3️⃣ Checking IP Address...');
    try {
        const response = await axios.get('https://api.ipify.org?format=json');
        console.log('Your IP:', response.data.ip);
        console.log('⚠️  Ensure this IP is whitelisted in Delta Exchange dashboard');
    } catch (error) {
        console.log('Could not determine IP:', error.message);
    }
}

async function main() {
    console.log('Configuration:');
    console.log('- API Key:', API_KEY);
    console.log('- API Secret Length:', API_SECRET.length);
    console.log('- Base URL:', BASE_URL);

    const publicApiWorking = await testPublicEndpoints();
    if (!publicApiWorking) {
        console.log('\n❌ Cannot proceed - Public API not accessible');
        return;
    }

    const authWorking = await testAuthEndpoints();
    if (!authWorking) {
        await checkIP();
        console.log('\n📋 Troubleshooting Steps:');
        console.log('1. Verify API key is copied correctly');
        console.log('2. Check IP whitelist in Delta Exchange dashboard');
        console.log('3. Ensure testnet account is properly set up');
        console.log('4. Try regenerating API keys');
    }
}

main().catch(console.error);
