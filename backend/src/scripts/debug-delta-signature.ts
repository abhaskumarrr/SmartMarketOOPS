/**
 * Debug Delta Exchange Signature Generation
 * Compare our signature generation with the working test
 */

import axios from 'axios';
import crypto from 'crypto';

// Load environment variables
require('dotenv').config({ path: require('path').resolve(__dirname, '../../../.env') });

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY || '';
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET || '';
const BASE_URL = 'https://cdn-ind.testnet.deltaex.org';

console.log('🔍 Debugging Delta Exchange Signature Generation...\n');

function generateSignature(
  method: string,
  path: string,
  queryString: string,
  body: string,
  timestamp: string
): string {
  const message = method + timestamp + path + queryString + body;
  console.log(`📝 Signature message: "${message}"`);
  
  const signature = crypto
    .createHmac('sha256', API_SECRET)
    .update(message)
    .digest('hex');
  
  console.log(`✍️ Generated signature: ${signature}`);
  return signature;
}

async function testBothMethods() {
  try {
    const method = 'GET';
    const path = '/v2/profile';
    const queryString = '';
    const body = '';
    const timestamp = Math.floor(Date.now() / 1000).toString();
    
    console.log(`⏰ Timestamp: ${timestamp}\n`);

    // Method 1: Working simple test approach
    console.log('🟢 Method 1: Working Simple Test Approach');
    console.log('=' .repeat(50));
    
    const signature1 = generateSignature(method, path, queryString, body, timestamp);
    
    const headers1 = {
      'api-key': API_KEY,
      'signature': signature1,
      'timestamp': timestamp,
      'Content-Type': 'application/json',
      'User-Agent': 'SmartMarketOOPS-TestBot-v1.0'
    };
    
    console.log('📤 Headers 1:', JSON.stringify(headers1, null, 2));
    
    try {
      const response1 = await axios.get(`${BASE_URL}${path}`, { headers: headers1 });
      console.log('✅ Method 1 SUCCESS:', response1.data.success);
    } catch (error: any) {
      console.log('❌ Method 1 FAILED:', error.response?.status, error.response?.data?.error || error.message);
    }
    
    console.log('\n');

    // Method 2: Unified service approach (simulate)
    console.log('🔴 Method 2: Unified Service Approach');
    console.log('=' .repeat(50));
    
    // Wait a moment to get a different timestamp
    await new Promise(resolve => setTimeout(resolve, 1000));
    const timestamp2 = Math.floor(Date.now() / 1000).toString();
    console.log(`⏰ New Timestamp: ${timestamp2}`);
    
    const signature2 = generateSignature(method, path, queryString, body, timestamp2);
    
    const headers2 = {
      'api-key': API_KEY,
      'signature': signature2,
      'timestamp': timestamp2,
      'Content-Type': 'application/json',
      'User-Agent': 'SmartMarketOOPS-DeltaBot-v1.0'
    };
    
    console.log('📤 Headers 2:', JSON.stringify(headers2, null, 2));
    
    try {
      // Use axios.request like the unified service does
      const response2 = await axios.request({
        method: method as any,
        url: `${BASE_URL}${path}${queryString}`,
        data: body || undefined,
        headers: headers2
      });
      console.log('✅ Method 2 SUCCESS:', response2.data.success);
    } catch (error: any) {
      console.log('❌ Method 2 FAILED:', error.response?.status, error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Method 3: Test with different endpoints
    console.log('🔵 Method 3: Test Different Endpoints');
    console.log('=' .repeat(50));
    
    const endpoints = [
      '/v2/profile',
      '/v2/wallet/balances',
      '/v2/positions'
    ];
    
    for (const endpoint of endpoints) {
      const timestamp3 = Math.floor(Date.now() / 1000).toString();
      const signature3 = generateSignature(method, endpoint, '', '', timestamp3);
      
      const headers3 = {
        'api-key': API_KEY,
        'signature': signature3,
        'timestamp': timestamp3,
        'Content-Type': 'application/json',
        'User-Agent': 'SmartMarketOOPS-TestBot-v1.0'
      };
      
      try {
        const response3 = await axios.get(`${BASE_URL}${endpoint}`, { headers: headers3 });
        console.log(`✅ ${endpoint}: SUCCESS`);
      } catch (error: any) {
        console.log(`❌ ${endpoint}: FAILED (${error.response?.status})`);
      }
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 500));
    }

  } catch (error) {
    console.error('💥 Test crashed:', error);
  }
}

// Run the debug test
testBothMethods()
  .then(() => {
    console.log('\n🏁 Debug test completed');
  })
  .catch((error) => {
    console.error('\n💥 Debug test failed:', error);
  });
