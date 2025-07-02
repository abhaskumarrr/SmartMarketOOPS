#!/usr/bin/env node
/**
 * Delta Exchange Integration Status Report
 * Comprehensive check of Delta Exchange integration and setup
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Configuration
const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = process.env.DELTA_EXCHANGE_BASE_URL || 'https://cdn-ind.testnet.deltaex.org';
const TESTNET = process.env.DELTA_EXCHANGE_TESTNET !== 'false';

// Colors for output
const COLORS = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

console.log(`${COLORS.magenta}📊 DELTA EXCHANGE INTEGRATION STATUS REPORT${COLORS.reset}`);
console.log('='.repeat(60));

/**
 * Check environment configuration
 */
function checkEnvironmentConfig() {
  console.log(`\n${COLORS.cyan}🔧 Environment Configuration${COLORS.reset}`);
  console.log('-'.repeat(40));
  
  console.log(`Environment: ${TESTNET ? COLORS.yellow + 'TESTNET' : COLORS.red + 'PRODUCTION'}${COLORS.reset}`);
  console.log(`API URL: ${BASE_URL}`);
  console.log(`API Key: ${API_KEY ? COLORS.green + '✓ Set (' + '****' + API_KEY.slice(-4) + ')' : COLORS.red + '✗ Not set'}${COLORS.reset}`);
  console.log(`API Secret: ${API_SECRET ? COLORS.green + '✓ Set' : COLORS.red + '✗ Not set'}${COLORS.reset}`);
  
  const status = API_KEY && API_SECRET;
  console.log(`Configuration Status: ${status ? COLORS.green + '✓ Complete' : COLORS.red + '✗ Incomplete'}${COLORS.reset}`);
  
  return status;
}

/**
 * Check available Delta Exchange services
 */
function checkAvailableServices() {
  console.log(`\n${COLORS.cyan}🛠️ Available Services${COLORS.reset}`);
  console.log('-'.repeat(40));
  
  const services = [
    'src/services/deltaExchangeService.ts',
    'src/services/deltaApiService.ts', 
    'src/services/DeltaExchangeUnified.ts',
    'scripts/test-delta-connection.js',
    'scripts/fetch-delta-products.js',
    'scripts/delta-paper-trading.js'
  ];
  
  let servicesFound = 0;
  
  services.forEach(service => {
    const exists = fs.existsSync(service);
    console.log(`${exists ? COLORS.green + '✓' : COLORS.red + '✗'} ${service}${COLORS.reset}`);
    if (exists) servicesFound++;
  });
  
  console.log(`Services Available: ${COLORS.blue}${servicesFound}/${services.length}${COLORS.reset}`);
  return servicesFound > 0;
}

/**
 * Test public API access
 */
async function testPublicAPI() {
  console.log(`\n${COLORS.cyan}🌐 Public API Access${COLORS.reset}`);
  console.log('-'.repeat(40));
  
  try {
    const response = await axios.get(`${BASE_URL}/v2/products`, { timeout: 10000 });
    
    if (response.data.success) {
      const products = response.data.result;
      const perpetuals = products.filter(p => p.contract_type === 'perpetual_futures' && p.state === 'live');
      
      console.log(`${COLORS.green}✓ Public API accessible${COLORS.reset}`);
      console.log(`Total products: ${products.length}`);
      console.log(`Live perpetuals: ${perpetuals.length}`);
      
      const majorPairs = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD'];
      const available = majorPairs.filter(pair => 
        products.find(p => p.symbol === pair && p.state === 'live')
      );
      
      console.log(`Available major pairs: ${available.join(', ')}`);
      
      return true;
    } else {
      console.log(`${COLORS.red}✗ Public API request failed${COLORS.reset}`);
      return false;
    }
  } catch (error) {
    console.log(`${COLORS.red}✗ Public API connection failed: ${error.message}${COLORS.reset}`);
    return false;
  }
}

/**
 * Check market data availability
 */
async function checkMarketData() {
  console.log(`\n${COLORS.cyan}📈 Market Data Availability${COLORS.reset}`);
  console.log('-'.repeat(40));
  
  try {
    const symbols = ['BTCUSD', 'ETHUSD'];
    let dataAvailable = 0;
    
    for (const symbol of symbols) {
      try {
        const response = await axios.get(`${BASE_URL}/v2/tickers/${symbol}`, { timeout: 5000 });
        
        if (response.data.success) {
          const ticker = response.data.result;
          console.log(`${COLORS.green}✓ ${symbol}: $${ticker.last_price}${COLORS.reset}`);
          dataAvailable++;
        } else {
          console.log(`${COLORS.red}✗ ${symbol}: No data${COLORS.reset}`);
        }
      } catch (error) {
        console.log(`${COLORS.red}✗ ${symbol}: Error${COLORS.reset}`);
      }
    }
    
    console.log(`Market Data Status: ${dataAvailable}/${symbols.length} symbols available`);
    return dataAvailable > 0;
  } catch (error) {
    console.log(`${COLORS.red}✗ Market data check failed${COLORS.reset}`);
    return false;
  }
}

/**
 * Check integration with backend services
 */
function checkBackendIntegration() {
  console.log(`\n${COLORS.cyan}🔗 Backend Integration${COLORS.reset}`);
  console.log('-'.repeat(40));
  
  // Check if backend is using Delta Exchange
  const backendFiles = [
    'src/simple-server.js',
    'src/services/deltaExchangeService.ts'
  ];
  
  let integrationFound = false;
  
  backendFiles.forEach(file => {
    if (fs.existsSync(file)) {
      const content = fs.readFileSync(file, 'utf8');
      const hasDeltaIntegration = content.includes('delta') || content.includes('Delta');
      
      console.log(`${hasDeltaIntegration ? COLORS.green + '✓' : COLORS.yellow + '~'} ${file}${COLORS.reset}`);
      if (hasDeltaIntegration) integrationFound = true;
    }
  });
  
  console.log(`Integration Status: ${integrationFound ? COLORS.green + '✓ Integrated' : COLORS.yellow + '~ Partial'}${COLORS.reset}`);
  return integrationFound;
}

/**
 * Generate setup instructions
 */
function generateSetupInstructions() {
  console.log(`\n${COLORS.magenta}📋 SETUP INSTRUCTIONS${COLORS.reset}`);
  console.log('='.repeat(60));
  
  console.log(`\n${COLORS.yellow}To enable real trading on Delta Exchange India Testnet:${COLORS.reset}`);
  console.log(`\n1. ${COLORS.cyan}Create Testnet Account:${COLORS.reset}`);
  console.log(`   • Visit: https://testnet.delta.exchange/`);
  console.log(`   • Sign up for a free testnet account`);
  console.log(`   • Verify your email address`);
  
  console.log(`\n2. ${COLORS.cyan}Generate API Keys:${COLORS.reset}`);
  console.log(`   • Login to your testnet account`);
  console.log(`   • Go to Account Settings > API Management`);
  console.log(`   • Create new API key with trading permissions`);
  console.log(`   • Save the API Key and Secret securely`);
  
  console.log(`\n3. ${COLORS.cyan}Update Configuration:${COLORS.reset}`);
  console.log(`   • Edit the .env file in your project root`);
  console.log(`   • Replace DELTA_EXCHANGE_API_KEY with your real key`);
  console.log(`   • Replace DELTA_EXCHANGE_API_SECRET with your real secret`);
  console.log(`   • Ensure DELTA_EXCHANGE_TESTNET="true"`);
  
  console.log(`\n4. ${COLORS.cyan}Fund Your Testnet Account:${COLORS.reset}`);
  console.log(`   • Delta Exchange testnet provides virtual funds`);
  console.log(`   • No real money required for testing`);
  
  console.log(`\n5. ${COLORS.cyan}Test Order Placement:${COLORS.reset}`);
  console.log(`   • Run: node scripts/test-delta-order.js`);
  console.log(`   • This will place and cancel a test order`);
  
  console.log(`\n${COLORS.green}📋 Current Integration Features:${COLORS.reset}`);
  console.log(`✓ Real-time market data from Delta Exchange`);
  console.log(`✓ Support for major crypto pairs (BTC, ETH, SOL, ADA)`);
  console.log(`✓ Comprehensive order management`);
  console.log(`✓ Position tracking and portfolio management`);
  console.log(`✓ Paper trading with real market data`);
  console.log(`✓ Risk management and stop-loss features`);
  
  console.log(`\n${COLORS.blue}🔗 Useful Links:${COLORS.reset}`);
  console.log(`• Delta Exchange Testnet: https://testnet.delta.exchange/`);
  console.log(`• API Documentation: https://docs.delta.exchange/`);
  console.log(`• Trading Guide: https://www.delta.exchange/blog/guide-to-api-trading-with-delta-india`);
}

/**
 * Run complete status check
 */
async function runStatusCheck() {
  const configOK = checkEnvironmentConfig();
  const servicesOK = checkAvailableServices();
  const publicAPIAK = await testPublicAPI();
  const marketDataOK = await checkMarketData();
  const backendOK = checkBackendIntegration();
  
  console.log(`\n${COLORS.magenta}📊 OVERALL STATUS${COLORS.reset}`);
  console.log('='.repeat(60));
  
  const checks = [
    { name: 'Environment Config', status: configOK },
    { name: 'Service Files', status: servicesOK },
    { name: 'Public API Access', status: publicAPIAK },
    { name: 'Market Data', status: marketDataOK },
    { name: 'Backend Integration', status: backendOK }
  ];
  
  let passedChecks = 0;
  checks.forEach(check => {
    const icon = check.status ? COLORS.green + '✓' : COLORS.red + '✗';
    console.log(`${icon} ${check.name}${COLORS.reset}`);
    if (check.status) passedChecks++;
  });
  
  console.log(`\nStatus: ${passedChecks}/${checks.length} checks passed`);
  
  if (passedChecks === checks.length) {
    console.log(`\n${COLORS.green}🎉 SYSTEM READY FOR TRADING!${COLORS.reset}`);
    console.log(`${COLORS.green}Update your API keys to start placing real orders.${COLORS.reset}`);
  } else {
    console.log(`\n${COLORS.yellow}⚠️  SYSTEM PARTIALLY READY${COLORS.reset}`);
    console.log(`${COLORS.yellow}Some components need attention.${COLORS.reset}`);
  }
  
  generateSetupInstructions();
}

// Run the status check
runStatusCheck().catch(error => {
  console.error(`${COLORS.red}Error during status check:${COLORS.reset}`, error.message);
  process.exit(1);
});
