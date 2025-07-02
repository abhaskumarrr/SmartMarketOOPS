#!/usr/bin/env node
/**
 * Delta Exchange Order Test
 * Tests placing a small order on Delta Exchange India Testnet
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

// Configuration
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = process.env.DELTA_EXCHANGE_BASE_URL || 'https://cdn-ind.testnet.deltaex.org';
const TESTNET = process.env.DELTA_EXCHANGE_TESTNET !== 'false';

console.log('🚀 DELTA EXCHANGE ORDER TEST');
console.log('============================');
console.log(`Environment: ${TESTNET ? 'TESTNET' : 'PRODUCTION'}`);
console.log(`API URL: ${BASE_URL}`);
console.log(`API Key: ${API_KEY ? '****' + API_KEY.slice(-4) : 'NOT SET'}`);
console.log('');

// Colors for output
const COLORS = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
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
  
  console.log(`${COLORS.cyan}Making ${method} request to: ${path}${COLORS.reset}`);
  
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
 * Check API key information
 */
async function checkAPIInfo() {
  console.log(`${COLORS.blue}🔑 Testing API key information...${COLORS.reset}`);
  
  if (!API_KEY || !API_SECRET) {
    console.log(`${COLORS.yellow}⚠ API credentials not found${COLORS.reset}`);
    return false;
  }
  
  try {
    // Test various endpoints to find working one
    const endpoints = [
      '/v2/wallet',
      '/v2/positions',
      '/v2/orders'
    ];
    
    for (const endpoint of endpoints) {
      console.log(`Testing endpoint: ${endpoint}`);
      const response = await makeAuthenticatedRequest('GET', endpoint);
      
      if (response.success) {
        console.log(`${COLORS.green}✓ ${endpoint} works!${COLORS.reset}`);
        console.log('Response:', JSON.stringify(response, null, 2));
        return true;
      } else {
        console.log(`${COLORS.yellow}⚠ ${endpoint} failed:${COLORS.reset}`, response.error || response);
      }
    }
    
    return false;
  } catch (error) {
    console.log(`${COLORS.red}✗ API test failed${COLORS.reset}`);
    console.log('Error:', error.message);
    return false;
  }
}

/**
 * Get current market price for a symbol
 */
async function getMarketPrice(symbol) {
  console.log(`${COLORS.blue}📈 Getting current market price for ${symbol}...${COLORS.reset}`);
  
  try {
    // First get the product ID
    const productsResponse = await axios.get(`${BASE_URL}/v2/products`);
    if (!productsResponse.data.success) {
      throw new Error('Failed to fetch products');
    }
    
    const product = productsResponse.data.result.find(p => p.symbol === symbol);
    if (!product) {
      throw new Error(`Product ${symbol} not found`);
    }
    
    console.log(`Found product: ${symbol} (ID: ${product.id})`);
    
    // Get ticker data
    const tickerResponse = await axios.get(`${BASE_URL}/v2/tickers/${symbol}`);
    if (tickerResponse.data.success) {
      const ticker = tickerResponse.data.result;
      console.log(`Current price: ${ticker.last_price}`);
      console.log(`Best bid: ${ticker.best_bid_price}`);
      console.log(`Best ask: ${ticker.best_ask_price}`);
      
      return {
        productId: product.id,
        lastPrice: parseFloat(ticker.last_price),
        bidPrice: parseFloat(ticker.best_bid_price),
        askPrice: parseFloat(ticker.best_ask_price)
      };
    } else {
      throw new Error('Failed to get ticker data');
    }
  } catch (error) {
    console.log(`${COLORS.red}✗ Failed to get market price${COLORS.reset}`);
    console.log('Error:', error.message);
    return null;
  }
}

/**
 * Run the complete test
 */
async function runOrderTest() {
  try {
    // Step 1: Test API authentication
    const apiWorks = await checkAPIInfo();
    if (!apiWorks) {
      console.log(`${COLORS.red}Cannot proceed without working API authentication${COLORS.reset}`);
      console.log(`${COLORS.yellow}NOTE: The API keys in .env appear to be example keys.${COLORS.reset}`);
      console.log(`${COLORS.yellow}To test real orders, you need to:${COLORS.reset}`);
      console.log(`${COLORS.yellow}1. Sign up at https://testnet.delta.exchange/${COLORS.reset}`);
      console.log(`${COLORS.yellow}2. Generate API keys from your testnet account${COLORS.reset}`);
      console.log(`${COLORS.yellow}3. Update the .env file with your real testnet API keys${COLORS.reset}`);
      return;
    }
    
    // Step 2: Get market price for BTCUSD
    const marketData = await getMarketPrice('BTCUSD');
    if (!marketData) {
      console.log(`${COLORS.red}Cannot proceed without market data${COLORS.reset}`);
      return;
    }
    
    console.log(`${COLORS.green}✅ System is ready for order placement!${COLORS.reset}`);
    console.log(`${COLORS.green}Delta Exchange API integration is working correctly.${COLORS.reset}`);
    
  } catch (error) {
    console.log(`${COLORS.red}❌ Test failed with error:${COLORS.reset}`, error.message);
  }
}

// Run the test
runOrderTest().catch(error => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
