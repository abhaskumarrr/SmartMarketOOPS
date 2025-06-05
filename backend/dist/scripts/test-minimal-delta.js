"use strict";
/**
 * Minimal Delta Exchange Test
 * Test the exact same approach as the unified service
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
console.log('🧪 Minimal Delta Exchange Test...\n');
function generateSignature(method, path, queryString, body, timestamp) {
    const message = method + timestamp + path + queryString + body;
    return crypto_1.default
        .createHmac('sha256', API_SECRET)
        .update(message)
        .digest('hex');
}
async function testMinimal() {
    try {
        // Create axios client like the unified service
        const client = axios_1.default.create({
            baseURL: BASE_URL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const method = 'GET';
        const path = '/v2/profile';
        const queryString = '';
        const body = '';
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const signature = generateSignature(method, path, queryString, body, timestamp);
        const headers = {
            'api-key': API_KEY,
            'signature': signature,
            'timestamp': timestamp,
            'Content-Type': 'application/json',
            'User-Agent': 'SmartMarketOOPS-DeltaBot-v1.0'
        };
        console.log(`🔍 Making request: ${method} ${path}${queryString}`);
        console.log(`📝 Signature message: "${method}${timestamp}${path}${queryString}${body}"`);
        console.log(`✍️ Generated signature: ${signature}`);
        console.log(`📤 Request headers:`, JSON.stringify(headers, null, 2));
        // Use the exact same request format as unified service
        const response = await client.request({
            method: method,
            url: path + queryString,
            data: undefined, // No data for GET request
            headers
        });
        console.log('✅ SUCCESS:', response.data.success);
        console.log('📊 Response:', JSON.stringify(response.data, null, 2));
    }
    catch (error) {
        console.error('❌ FAILED:');
        if (error.response) {
            console.error(`Status: ${error.response.status}`);
            console.error(`Data:`, error.response.data);
        }
        else {
            console.error('Error:', error.message);
        }
    }
}
// Run the test
testMinimal()
    .then(() => {
    console.log('\n🏁 Test completed');
})
    .catch((error) => {
    console.error('\n💥 Test failed:', error);
});
//# sourceMappingURL=test-minimal-delta.js.map