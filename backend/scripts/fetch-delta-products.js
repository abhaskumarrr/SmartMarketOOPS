#!/usr/bin/env node
/**
 * Delta Exchange Product ID Fetcher
 * 
 * This script fetches all available products from Delta Exchange API
 * and outputs them in a format suitable for updating the codebase.
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

// Constants
const DELTA_BASE_URL = process.env.DELTA_EXCHANGE_BASE_URL || 'https://cdn-ind.testnet.deltaex.org';
const DELTA_API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const DELTA_API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;

// Hardcoded product IDs as fallback
const FALLBACK_PRODUCT_IDS = {
  testnet: {
    'BTCUSD': 84,
    'ETHUSD': 1699,
    'SOLUSD': 84401,
    'BNBUSD': 84402,
    'ADAUSD': 84403
  },
  production: {
    'BTCUSD': 27,
    'ETHUSD': 3136,
    'SOLUSD': 52001,
    'BNBUSD': 52002,
    'ADAUSD': 52003
  }
};

/**
 * Generate signature for Delta Exchange API
 */
function generateSignature(method, path, queryString, body, timestamp) {
  const message = method + timestamp + path + queryString + (body || '');
  return crypto
    .createHmac('sha256', DELTA_API_SECRET)
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
    'X-DELTA-API-KEY': DELTA_API_KEY,
    'X-DELTA-SIGNATURE': signature,
    'X-DELTA-TIMESTAMP': timestamp
  };
  
  const url = `${DELTA_BASE_URL}${path}${queryString}`;
  
  const response = await axios({
    method,
    url,
    headers,
    data: body ? JSON.parse(body) : undefined
  });
  
  return response.data;
}

/**
 * Fetch products from Delta Exchange API
 */
async function fetchProducts(baseUrl, isTestnet = true) {
  try {
    console.log(`Fetching products from ${isTestnet ? 'TESTNET' : 'PRODUCTION'} API: ${baseUrl}`);
    
    let products = [];
    
    try {
      // Try unauthenticated request first
      const response = await axios.get(`${baseUrl}/v2/products`);
      if (response.data.success) {
        products = response.data.result;
      } else {
        throw new Error('Unauthenticated request failed');
      }
    } catch (error) {
      // If unauthenticated fails, try authenticated request
      console.log('Unauthenticated request failed, trying with API key...');
      if (DELTA_API_KEY && DELTA_API_SECRET) {
        try {
          const authResponse = await makeAuthenticatedRequest('GET', '/v2/products');
          if (authResponse.success) {
            products = authResponse.result;
          } else {
            throw new Error('Authentication failed');
          }
        } catch (authError) {
          console.error('Authenticated request failed:', authError.message);
          throw authError;
        }
      } else {
        console.error('No API credentials available');
        throw error;
      }
    }
    
    // Filter for perpetual futures and important products
    const perpetuals = products.filter(p => 
      p.product_type === 'perpetual_futures' || 
      ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD', 'ADAUSD'].includes(p.symbol)
    );
    
    console.log(`\n${isTestnet ? 'TESTNET' : 'PRODUCTION'} PERPETUAL PRODUCTS:`);
    console.log('-------------------------------------------------------------------------');
    console.log('PRODUCT_ID | SYMBOL  | STATE    | CONTRACT_SIZE | MAINTENANCE_MARGIN');
    console.log('-------------------------------------------------------------------------');
    
    perpetuals.forEach(p => {
      console.log(`${p.id.toString().padEnd(10)} | ${p.symbol.padEnd(7)} | ${(p.state || 'unknown').padEnd(8)} | ${(p.contract_size || 'N/A').toString().padEnd(13)} | ${p.maintenance_margin || 'N/A'}%`);
    });
    
    console.log('\nFOR .ENV CONFIG:');
    console.log(`DELTA_BTCUSD_PRODUCT_ID=${perpetuals.find(p => p.symbol === 'BTCUSD')?.id || 'NOT_FOUND'}`);
    console.log(`DELTA_ETHUSD_PRODUCT_ID=${perpetuals.find(p => p.symbol === 'ETHUSD')?.id || 'NOT_FOUND'}`);
    console.log(`DELTA_SOLUSD_PRODUCT_ID=${perpetuals.find(p => p.symbol === 'SOLUSD')?.id || 'NOT_FOUND'}`);
    
    return perpetuals;
  } catch (error) {
    console.error('Error fetching products:', error.message);
    
    // Use fallback product IDs
    console.log('\nUsing fallback product IDs:');
    console.log('-------------------------------------------------------------------------');
    console.log('PRODUCT_ID | SYMBOL  | STATE    | CONTRACT_SIZE | MAINTENANCE_MARGIN');
    console.log('-------------------------------------------------------------------------');
    
    const fallbackIds = isTestnet ? FALLBACK_PRODUCT_IDS.testnet : FALLBACK_PRODUCT_IDS.production;
    
    Object.entries(fallbackIds).forEach(([symbol, id]) => {
      console.log(`${id.toString().padEnd(10)} | ${symbol.padEnd(7)} | live     | N/A          | N/A%`);
    });
    
    console.log('\nFOR .ENV CONFIG (FALLBACK VALUES):');
    console.log(`DELTA_BTCUSD_PRODUCT_ID=${fallbackIds['BTCUSD']}`);
    console.log(`DELTA_ETHUSD_PRODUCT_ID=${fallbackIds['ETHUSD']}`);
    console.log(`DELTA_SOLUSD_PRODUCT_ID=${fallbackIds['SOLUSD']}`);
    
    return [];
  }
}

async function main() {
  console.log('🚀 DELTA EXCHANGE PRODUCT ID FETCHER');
  console.log('===================================\n');
  
  // Fetch from testnet
  await fetchProducts(DELTA_BASE_URL, true);
  
  console.log('\n✅ PRODUCT ID UPDATE INSTRUCTIONS:');
  console.log('1. Update your .env file with the IDs shown above');
  console.log('2. Update the product IDs in your trading bot configuration files');
  console.log('3. Search the codebase for hardcoded product IDs and replace them');
  console.log('\nFiles to update:');
  console.log('- backend/scripts/delta-testnet-live.js');
  console.log('- backend/scripts/delta-testnet-trading.js');
  console.log('- backend/src/routes/tradingRoutesWorking.ts');
  console.log('- backend/src/services/trading/apiKeyValidationService.ts');
  console.log('- example.env and .env');
}

main().catch(console.error); 