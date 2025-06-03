#!/usr/bin/env node
"use strict";
/**
 * Debug Delta API - Check actual response formats
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = __importDefault(require("axios"));
async function debugDeltaAPI() {
    console.log('🔍 DEBUGGING DELTA API RESPONSES');
    console.log('='.repeat(60));
    const testnetUrl = 'https://testnet-api.delta.exchange';
    try {
        // Test 1: Check if server is reachable
        console.log('📡 Testing basic connectivity...');
        const response = await axios_1.default.get(`${testnetUrl}/v2/products`, {
            timeout: 10000
        });
        console.log(`✅ Server reachable: ${response.status}`);
        console.log(`📊 Response type: ${typeof response.data}`);
        console.log(`📊 Response keys:`, Object.keys(response.data || {}));
        if (response.data) {
            console.log('📊 Sample response structure:');
            console.log(JSON.stringify(response.data, null, 2).substring(0, 500) + '...');
        }
        // Test 2: Check different endpoints
        const endpoints = [
            '/v1/time',
            '/v2/time',
            '/v1/products',
            '/v2/products'
        ];
        for (const endpoint of endpoints) {
            try {
                console.log(`\n🔍 Testing endpoint: ${endpoint}`);
                const endpointResponse = await axios_1.default.get(`${testnetUrl}${endpoint}`, {
                    timeout: 5000
                });
                console.log(`✅ ${endpoint}: ${endpointResponse.status} - ${typeof endpointResponse.data}`);
                if (endpoint.includes('time')) {
                    console.log(`   Time data:`, endpointResponse.data);
                }
                else if (endpoint.includes('products')) {
                    const data = endpointResponse.data;
                    if (Array.isArray(data)) {
                        console.log(`   Products: ${data.length} items`);
                    }
                    else if (data && data.result) {
                        console.log(`   Products: ${Array.isArray(data.result) ? data.result.length : 'Not array'} items`);
                    }
                    else {
                        console.log(`   Products structure:`, Object.keys(data || {}));
                    }
                }
            }
            catch (error) {
                console.log(`❌ ${endpoint}: ${error.response?.status || error.message}`);
            }
        }
    }
    catch (error) {
        console.log('❌ Basic connectivity failed:', error.message);
        console.log('🔧 Check if Delta testnet is accessible');
    }
    console.log('\n🎉 Debug completed');
}
debugDeltaAPI().catch(console.error);
//# sourceMappingURL=debug-delta-api.js.map