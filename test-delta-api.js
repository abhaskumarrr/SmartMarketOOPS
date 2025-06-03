#!/usr/bin/env node

const axios = require('axios');

async function testDeltaExchangeAPI() {
  console.log('🧪 Testing Delta Exchange API Integration...\n');

  // Test 1: Get all products
  console.log('📋 Test 1: Fetching all products...');
  try {
    const response = await axios.get('https://cdn-ind.testnet.deltaex.org/v2/products');
    console.log(`✅ Status: ${response.status}`);
    
    if (response.data && response.data.success) {
      const products = response.data.result;
      console.log(`✅ Found ${products.length} products`);
      
      // Find BTCUSD and ETHUSD
      const btcProduct = products.find(p => p.symbol === 'BTCUSD');
      const ethProduct = products.find(p => p.symbol === 'ETHUSD');

      if (btcProduct) {
        console.log(`✅ BTCUSD found - ID: ${btcProduct.id}, Type: ${btcProduct.product_type}`);
        console.log('   Full product data:', JSON.stringify(btcProduct, null, 2));
      } else {
        console.log('❌ BTCUSD not found');
        // Show first few products to understand structure
        console.log('First 3 products:', products.slice(0, 3).map(p => ({ id: p.id, symbol: p.symbol, product_type: p.product_type })));
      }

      if (ethProduct) {
        console.log(`✅ ETHUSD found - ID: ${ethProduct.id}, Type: ${ethProduct.product_type}`);
        console.log('   Full product data:', JSON.stringify(ethProduct, null, 2));
      } else {
        console.log('❌ ETHUSD not found');
      }
    } else {
      console.log('❌ API returned success: false');
      console.log('Response:', response.data);
    }
  } catch (error) {
    console.log('❌ Error:', error.message);
    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }
  }

  console.log('\n' + '='.repeat(50) + '\n');

  // Test 2: Get ticker for BTCUSD using symbol
  console.log('📊 Test 2: Fetching BTCUSD ticker using symbol...');
  try {
    const response = await axios.get('https://cdn-ind.testnet.deltaex.org/v2/tickers/BTCUSD');
    console.log(`✅ Status: ${response.status}`);
    
    if (response.data && response.data.success) {
      const ticker = response.data.result;
      console.log('✅ BTCUSD Ticker Data:');
      console.log(`   Price: $${ticker.close || ticker.last_price || 'N/A'}`);
      console.log(`   Change: ${ticker.change || 'N/A'}`);
      console.log(`   Change %: ${ticker.change_percent || 'N/A'}%`);
      console.log(`   Volume: ${ticker.volume || 'N/A'}`);
      console.log(`   High: $${ticker.high || 'N/A'}`);
      console.log(`   Low: $${ticker.low || 'N/A'}`);
      console.log(`   Mark Price: $${ticker.mark_price || 'N/A'}`);
      console.log(`   Index Price: $${ticker.spot_price || 'N/A'}`);
    } else {
      console.log('❌ API returned success: false');
      console.log('Response:', response.data);
    }
  } catch (error) {
    console.log('❌ Error:', error.message);
    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }
  }

  console.log('\n' + '='.repeat(50) + '\n');

  // Test 3: Get ticker for ETHUSD using symbol
  console.log('📊 Test 3: Fetching ETHUSD ticker using symbol...');
  try {
    const response = await axios.get('https://cdn-ind.testnet.deltaex.org/v2/tickers/ETHUSD');
    console.log(`✅ Status: ${response.status}`);
    
    if (response.data && response.data.success) {
      const ticker = response.data.result;
      console.log('✅ ETHUSD Ticker Data:');
      console.log(`   Price: $${ticker.close || ticker.last_price || 'N/A'}`);
      console.log(`   Change: ${ticker.change || 'N/A'}`);
      console.log(`   Change %: ${ticker.change_percent || 'N/A'}%`);
      console.log(`   Volume: ${ticker.volume || 'N/A'}`);
      console.log(`   High: $${ticker.high || 'N/A'}`);
      console.log(`   Low: $${ticker.low || 'N/A'}`);
      console.log(`   Mark Price: $${ticker.mark_price || 'N/A'}`);
      console.log(`   Index Price: $${ticker.spot_price || 'N/A'}`);
    } else {
      console.log('❌ API returned success: false');
      console.log('Response:', response.data);
    }
  } catch (error) {
    console.log('❌ Error:', error.message);
    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }
  }

  console.log('\n🏁 Delta Exchange API Test Complete!');
}

// Run the test
testDeltaExchangeAPI().catch(console.error);
