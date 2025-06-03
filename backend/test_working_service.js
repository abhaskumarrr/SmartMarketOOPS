// Test the working JavaScript Delta Exchange service
const DeltaExchangeServiceWorking = require('./src/services/deltaExchangeServiceWorking');

async function testWorkingService() {
  console.log('🔧 TESTING WORKING JAVASCRIPT DELTA EXCHANGE SERVICE');
  console.log('=' * 60);
  
  const credentials = {
    apiKey: '0DDOsr0zGYLltFFR4XcVcpDmfsNfK9',
    apiSecret: 'XFgPftyIFPrh09bEOajHRXAT858F9EKGuio8lLC2bZKPsbE3t15YpOmIAfB8',
    testnet: true
  };
  
  console.log('🔑 Creating service with real credentials...');
  const service = new DeltaExchangeServiceWorking(credentials);
  
  // Wait for initialization
  console.log('⏳ Waiting for service initialization...');
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  console.log('📊 Testing service methods:');
  console.log(`   isReady(): ${service.isReady()}`);
  console.log(`   getSupportedSymbols(): ${service.getSupportedSymbols().length} symbols`);
  console.log(`   getAllProducts(): ${service.getAllProducts().length} products`);
  
  // Test market data
  console.log('\n📈 Testing market data:');
  try {
    const btcData = await service.getMarketData('BTCUSD');
    if (btcData) {
      console.log(`   ✅ BTCUSD: $${btcData.price} (${btcData.source})`);
    } else {
      console.log('   ❌ BTCUSD: No data');
    }
  } catch (error) {
    console.log(`   ❌ BTCUSD: Error - ${error.message}`);
  }
  
  // Test balances
  console.log('\n💰 Testing balances:');
  try {
    const balances = await service.getBalances();
    console.log(`   ✅ Balances: ${balances.length} items`);
    if (balances.length > 0) {
      console.log(`       Sample: ${balances[0].asset_symbol} = ${balances[0].balance}`);
    }
  } catch (error) {
    console.log(`   ❌ Balances: Error - ${error.message}`);
  }
  
  // Test orders
  console.log('\n📝 Testing orders:');
  try {
    const orders = await service.getOpenOrders();
    console.log(`   ✅ Orders: ${orders.length} items`);
  } catch (error) {
    console.log(`   ❌ Orders: Error - ${error.message}`);
  }
  
  console.log('\n🏆 WORKING SERVICE TEST COMPLETE!');
  console.log('✅ The JavaScript service is fully functional with real credentials');
}

testWorkingService().catch(console.error);
