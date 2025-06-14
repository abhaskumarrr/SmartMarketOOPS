#!/usr/bin/env node
/**
 * Delta Exchange API Connection Test
 * 
 * This script tests the connection to the Delta Exchange API
 * and verifies that API keys are working correctly.
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

// Load configuration from environment
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = process.env.DELTA_EXCHANGE_BASE_URL || 'https://cdn-ind.testnet.deltaex.org';
const TESTNET = process.env.DELTA_EXCHANGE_TESTNET !== 'false';

// Constants
const COLORS = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

/**
 * Generate signature for Delta Exchange API
 */
function generateSignature(method, path, queryString, body, timestamp) {
  const message = method + timestamp + path + queryString + (body || '');
  return crypto
    .createHmac('sha256', API_SECRET)
    .update(message)
    .digest('hex');
}

/**
 * Make authenticated request to Delta Exchange API
 */
async function makeAuthenticatedRequest(method, path, params = {}, data = {}) {
  const timestamp = Date.now().toString();
  const queryString = Object.keys(params).length > 0 
    ? '?' + new URLSearchParams(params).toString() 
    : '';
  
  const body = method !== 'GET' && Object.keys(data).length > 0 
    ? JSON.stringify(data) 
    : '';
  
  const signature = generateSignature(
    method, 
    path, 
    queryString,
    body,
    timestamp
  );
  
  const headers = {
    'Content-Type': 'application/json',
    'X-DELTA-API-KEY': API_KEY,
    'X-DELTA-SIGNATURE': signature,
    'X-DELTA-TIMESTAMP': timestamp
  };
  
  const url = `${BASE_URL}${path}${queryString}`;
  
  try {
    const response = await axios({
      method,
      url,
      headers,
      data: body ? JSON.parse(body) : undefined
    });
    
    return response.data;
  } catch (error) {
    if (error.response) {
      return {
        success: false,
        error: error.response.data,
        status: error.response.status
      };
    } else {
      throw error;
    }
  }
}

/**
 * Test public API endpoint
 */
async function testPublicEndpoint() {
  console.log(`${COLORS.cyan}Testing public API endpoint...${COLORS.reset}`);
  
  try {
    const response = await axios.get(`${BASE_URL}/v2/products`);
    
    if (response.data.success) {
      console.log(`${COLORS.green}✓ Public API working correctly${COLORS.reset}`);
      const productCount = response.data.result.length;
      console.log(`  Found ${productCount} products`);
      return true;
    } else {
      console.log(`${COLORS.red}✗ Public API request failed${COLORS.reset}`);
      console.log('  Response:', response.data);
      return false;
    }
  } catch (error) {
    console.log(`${COLORS.red}✗ Public API request failed${COLORS.reset}`);
    console.log('  Error:', error.message);
    return false;
  }
}

/**
 * Test authenticated API endpoint
 */
async function testAuthenticatedEndpoint() {
  console.log(`${COLORS.cyan}Testing authenticated API endpoint...${COLORS.reset}`);
  
  if (!API_KEY || !API_SECRET) {
    console.log(`${COLORS.yellow}⚠ API credentials not found in .env file${COLORS.reset}`);
    console.log('  Set DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET in your .env file');
    return false;
  }
  
  try {
    const response = await makeAuthenticatedRequest('GET', '/v2/wallet');
    
    if (response.success) {
      console.log(`${COLORS.green}✓ Authentication working correctly${COLORS.reset}`);
      const wallets = response.result;
      console.log(`  Found ${wallets.length} wallets in your account`);
      
      // Print balance info
      if (wallets.length > 0) {
        console.log('  Available balances:');
        wallets.forEach(wallet => {
          if (parseFloat(wallet.available_balance) > 0) {
            console.log(`    ${wallet.currency_symbol}: ${wallet.available_balance}`);
          }
        });
      }
      
      return true;
    } else {
      console.log(`${COLORS.red}✗ Authentication failed${COLORS.reset}`);
      console.log('  Response:', response.error || response);
      return false;
    }
  } catch (error) {
    console.log(`${COLORS.red}✗ Authentication failed${COLORS.reset}`);
    console.log('  Error:', error.message);
    return false;
  }
}

/**
 * Run all tests
 */
async function runTests() {
  console.log(`${COLORS.magenta}DELTA EXCHANGE API CONNECTION TEST${COLORS.reset}`);
  console.log(`${COLORS.magenta}=================================${COLORS.reset}\n`);
  
  console.log(`${COLORS.blue}Environment:${COLORS.reset} ${TESTNET ? 'TESTNET' : 'PRODUCTION'}`);
  console.log(`${COLORS.blue}API URL:${COLORS.reset} ${BASE_URL}`);
  console.log(`${COLORS.blue}API Key:${COLORS.reset} ${API_KEY ? '****' + API_KEY.slice(-4) : 'NOT SET'}`);
  console.log(`${COLORS.blue}API Secret:${COLORS.reset} ${API_SECRET ? '********' : 'NOT SET'}\n`);
  
  // Test public endpoint
  const publicResult = await testPublicEndpoint();
  
  // Test authenticated endpoint
  const authResult = await testAuthenticatedEndpoint();
  
  console.log(`\n${COLORS.magenta}TEST SUMMARY${COLORS.reset}`);
  console.log(`${COLORS.blue}Public API:${COLORS.reset} ${publicResult ? COLORS.green + 'PASS' : COLORS.red + 'FAIL'}`);
  console.log(`${COLORS.blue}Authentication:${COLORS.reset} ${authResult ? COLORS.green + 'PASS' : COLORS.red + 'FAIL'}`);
  console.log(COLORS.reset);
  
  if (publicResult && authResult) {
    console.log(`${COLORS.green}All tests passed! Your Delta Exchange API configuration is working correctly.${COLORS.reset}`);
    return true;
  } else {
    console.log(`${COLORS.yellow}Some tests failed. Please check your configuration.${COLORS.reset}`);
    return false;
  }
}

// Run tests
runTests().catch(error => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
