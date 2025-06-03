#!/usr/bin/env node

const axios = require('axios');

async function testBackendAPI() {
  console.log('🧪 Testing Backend API Integration...\n');

  const baseURL = 'http://localhost:3001';

  // Test 1: Paper Trading Market Data
  console.log('📊 Test 1: Testing Paper Trading Market Data API...');
  try {
    const response = await axios.get(`${baseURL}/api/paper-trading/market-data`);
    console.log(`✅ Status: ${response.status}`);
    
    if (response.data && response.data.success) {
      const marketData = response.data.data;
      console.log('✅ Market Data Response:');
      
      for (const [symbol, data] of Object.entries(marketData)) {
        console.log(`   ${symbol}:`);
        console.log(`     Price: $${data.price}`);
        console.log(`     Change: ${data.change}`);
        console.log(`     Change %: ${data.changePercent}%`);
        console.log(`     Volume: ${data.volume}`);
        console.log(`     High: $${data.high24h}`);
        console.log(`     Low: $${data.low24h}`);
        console.log(`     Source: ${data.source}`);
        if (data.markPrice) console.log(`     Mark Price: $${data.markPrice}`);
        if (data.indexPrice) console.log(`     Index Price: $${data.indexPrice}`);
        console.log('');
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

  // Test 2: Portfolio Data
  console.log('💼 Test 2: Testing Portfolio API...');
  try {
    const response = await axios.get(`${baseURL}/api/paper-trading/portfolio`);
    console.log(`✅ Status: ${response.status}`);
    
    if (response.data && response.data.success) {
      const portfolio = response.data.data;
      console.log('✅ Portfolio Data:');
      console.log(`   Balance: $${portfolio.balance}`);
      console.log(`   Total Value: $${portfolio.totalValue}`);
      console.log(`   P&L: $${portfolio.totalPnL}`);
      console.log(`   Positions: ${portfolio.positions.length}`);
      
      if (portfolio.positions.length > 0) {
        console.log('   Open Positions:');
        portfolio.positions.forEach((pos, i) => {
          console.log(`     ${i + 1}. ${pos.symbol}: ${pos.quantity} @ $${pos.entryPrice}`);
        });
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

  // Test 3: Health Check
  console.log('🏥 Test 3: Testing Health Check...');
  try {
    const response = await axios.get(`${baseURL}/health`);
    console.log(`✅ Status: ${response.status}`);
    console.log('✅ Health Check Response:', response.data);
  } catch (error) {
    console.log('❌ Error:', error.message);
    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response data:', error.response.data);
    }
  }

  console.log('\n🏁 Backend API Test Complete!');
}

// Check if server is running first
async function checkServerStatus() {
  try {
    await axios.get('http://localhost:3001/health', { timeout: 2000 });
    return true;
  } catch (error) {
    return false;
  }
}

// Main execution
async function main() {
  console.log('🔍 Checking if backend server is running...');
  
  const isRunning = await checkServerStatus();
  
  if (!isRunning) {
    console.log('❌ Backend server is not running on port 3001');
    console.log('💡 Please start the backend server first:');
    console.log('   cd backend && npm start');
    process.exit(1);
  }
  
  console.log('✅ Backend server is running!\n');
  await testBackendAPI();
}

main().catch(console.error);
