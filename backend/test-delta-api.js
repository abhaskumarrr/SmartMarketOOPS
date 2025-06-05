const crypto = require('crypto');
const axios = require('axios');

// Test Delta Exchange API credentials from .env file
const API_KEY = 'AjTdJYCVE3aMZDAVQ2r6AQdmkU2mWc';
const API_SECRET = 'R29RkXJfUIIt4o3vCDXImyg6q74JvByYltVKFH96UJG51lR1mm88PCGnMrUR';
const BASE_URL = 'https://cdn-ind.testnet.deltaex.org';

function generateSignature(method, path, queryString, body, timestamp, secret) {
    const message = method + timestamp + path + queryString + body;
    console.log('Signature components:');
    console.log('  Method:', method);
    console.log('  Timestamp:', timestamp);
    console.log('  Path:', path);
    console.log('  QueryString:', queryString);
    console.log('  Body:', body);
    console.log('  Message:', message);
    const signature = crypto.createHmac('sha256', secret).update(message).digest('hex');
    console.log('  Signature:', signature);
    return signature;
}

async function testAPI() {
    console.log('🔍 Testing Delta Exchange API...');
    console.log('Base URL:', BASE_URL);
    console.log('API Key:', API_KEY);
    console.log('Current IP should be whitelisted: 122.177.255.130');

    try {
        // Test 1: Public endpoint (no auth required)
        console.log('\n📊 Test 1: Public Assets Endpoint');
        const publicResponse = await axios.get(`${BASE_URL}/v2/assets`);
        console.log('✅ Public API works:', publicResponse.data.success);
        console.log('Assets count:', publicResponse.data.result?.length || 0);

        // Test 2: Try different base URLs
        const testUrls = [
            'https://cdn-ind.testnet.deltaex.org',
            'https://testnet-api.delta.exchange',
            'https://api.testnet.delta.exchange'
        ];

        for (const testUrl of testUrls) {
            console.log(`\n🔐 Testing authenticated endpoint with: ${testUrl}`);

            const timestamp = Math.floor(Date.now() / 1000).toString();
            const method = 'GET';
            const path = '/v2/profile';
            const queryString = '';
            const body = '';

            const signature = generateSignature(method, path, queryString, body, timestamp, API_SECRET);

            const headers = {
                'api-key': API_KEY,
                'signature': signature,
                'timestamp': timestamp,
                'User-Agent': 'SmartMarketOOPS-Test',
                'Content-Type': 'application/json'
            };

            try {
                const authResponse = await axios.get(`${testUrl}${path}`, { headers });
                console.log(`✅ ${testUrl} works:`, authResponse.data.success);
                console.log('Profile data:', JSON.stringify(authResponse.data, null, 2));

                // If this URL works, test balance endpoint
                console.log(`\n💰 Testing balance endpoint with: ${testUrl}`);
                const balanceTimestamp = Math.floor(Date.now() / 1000).toString();
                const balancePath = '/v2/wallet/balances';
                const balanceSignature = generateSignature('GET', balancePath, '', '', balanceTimestamp, API_SECRET);

                const balanceHeaders = {
                    'api-key': API_KEY,
                    'signature': balanceSignature,
                    'timestamp': balanceTimestamp,
                    'User-Agent': 'SmartMarketOOPS-Test',
                    'Content-Type': 'application/json'
                };

                const balanceResponse = await axios.get(`${testUrl}${balancePath}`, { headers: balanceHeaders });
                console.log('✅ Balance API works:', balanceResponse.data.success);
                console.log('Balance data:', JSON.stringify(balanceResponse.data, null, 2));

                break; // If we found a working URL, stop testing others

            } catch (urlError) {
                console.log(`❌ ${testUrl} failed:`, urlError.response?.status, urlError.response?.data?.error?.code);
            }
        }

    } catch (error) {
        console.error('❌ API Test Failed:');
        if (error.response) {
            console.error('Status:', error.response.status);
            console.error('Headers:', error.response.headers);
            console.error('Data:', error.response.data);
        } else {
            console.error('Error:', error.message);
        }
    }
}

testAPI();
