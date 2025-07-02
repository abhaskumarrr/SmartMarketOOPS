#!/usr/bin/env node
/**
 * Final Delta Exchange Authentication Test
 * With correct timestamp format
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('🎯 DELTA EXCHANGE FINAL AUTH TEST');
console.log('================================');

async function testAuth() {
    const endpoint = '/v2/orders';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    // Using correct timestamp format (seconds, not milliseconds)
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
        return true;
    } catch (error) {
        console.log('\n❌ Error:', error.response?.data || error.message);
        return false;
    }
}

async function testWallet() {
    console.log('\n💰 Testing Wallet Access...');
    const endpoint = '/v2/wallet';
    const timestamp = Math.floor(Date.now() / 1000).toString();
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

    try {
        const response = await axios.get(`${BASE_URL}${endpoint}`, { headers });
        console.log('✅ Wallet accessible!');
        console.log(response.data);
        return true;
    } catch (error) {
        console.log('❌ Wallet error:', error.response?.data || error.message);
        return false;
    }
}

async function main() {
    const authSuccess = await testAuth();
    if (authSuccess) {
        await testWallet();
    } else {
        console.log('\n⚠️ Authentication Failed');
        console.log('Please check:');
        console.log('1. API key activation status');
        console.log('2. IP whitelist (49.37.8.175)');
        console.log('3. Testnet account setup');
    }
}

main().catch(console.error);
