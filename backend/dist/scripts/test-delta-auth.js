"use strict";
/**
 * Simple Delta Exchange Authentication Test
 * Test basic authentication with Delta Exchange India testnet
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = __importDefault(require("axios"));
const crypto_1 = __importDefault(require("crypto"));
// Load environment variables
require('dotenv').config({ path: require('path').resolve(__dirname, '../../../.env') });
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY || '';
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET || '';
const BASE_URL = 'https://cdn-ind.testnet.deltaex.org';
console.log('🔐 Testing Delta Exchange Authentication...\n');
if (!API_KEY || !API_SECRET) {
    console.error('❌ API credentials not found in environment variables');
    process.exit(1);
}
console.log(`🔑 API Key: ${API_KEY.substring(0, 8)}...${API_KEY.substring(API_KEY.length - 4)}`);
console.log(`🔒 API Secret: ${API_SECRET.substring(0, 8)}...${API_SECRET.substring(API_SECRET.length - 4)}`);
console.log(`🌐 Base URL: ${BASE_URL}\n`);
function generateSignature(method, path, queryString, body, timestamp) {
    const message = method + timestamp + path + queryString + body;
    console.log(`📝 Signature message: "${message}"`);
    const signature = crypto_1.default
        .createHmac('sha256', API_SECRET)
        .update(message)
        .digest('hex');
    console.log(`✍️ Generated signature: ${signature}`);
    return signature;
}
async function testAuthentication() {
    try {
        // Test 1: Public endpoint (no auth required)
        console.log('1. 🌍 Testing public endpoint...');
        const publicResponse = await axios_1.default.get(`${BASE_URL}/v2/products`);
        if (publicResponse.data.success) {
            console.log(`✅ Public API working - ${publicResponse.data.result.length} products found\n`);
        }
        else {
            console.log('❌ Public API failed\n');
            return;
        }
        // Test 2: Authenticated endpoint
        console.log('2. 🔐 Testing authenticated endpoint...');
        const method = 'GET';
        const path = '/v2/profile';
        const queryString = '';
        const body = '';
        const timestamp = Math.floor(Date.now() / 1000).toString();
        console.log(`⏰ Timestamp: ${timestamp}`);
        const signature = generateSignature(method, path, queryString, body, timestamp);
        const headers = {
            'api-key': API_KEY,
            'signature': signature,
            'timestamp': timestamp,
            'Content-Type': 'application/json',
            'User-Agent': 'SmartMarketOOPS-TestBot-v1.0'
        };
        console.log('📤 Request headers:', JSON.stringify(headers, null, 2));
        const authResponse = await axios_1.default.get(`${BASE_URL}${path}`, { headers });
        if (authResponse.data.success) {
            console.log('✅ Authentication successful!');
            console.log('👤 Profile data:', JSON.stringify(authResponse.data.result, null, 2));
        }
        else {
            console.log('❌ Authentication failed:', authResponse.data);
        }
    }
    catch (error) {
        console.error('❌ Test failed:');
        if (error.response) {
            console.error(`Status: ${error.response.status}`);
            console.error(`Headers:`, error.response.headers);
            console.error(`Data:`, error.response.data);
        }
        else if (error.request) {
            console.error('No response received:', error.request);
        }
        else {
            console.error('Error:', error.message);
        }
    }
}
// Run the test
testAuthentication()
    .then(() => {
    console.log('\n🏁 Test completed');
})
    .catch((error) => {
    console.error('\n💥 Test crashed:', error);
    process.exit(1);
});
//# sourceMappingURL=test-delta-auth.js.map