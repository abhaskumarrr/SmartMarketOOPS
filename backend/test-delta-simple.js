// Simple test using our existing Delta Exchange service
require('dotenv').config();

const { DeltaExchangeService } = require('./temp/deltaExchangeService');

async function testDeltaService() {
    console.log('🔍 Testing Delta Exchange Service...');
    
    const credentials = {
        apiKey: process.env.DELTA_EXCHANGE_API_KEY,
        apiSecret: process.env.DELTA_EXCHANGE_API_SECRET,
        testnet: true
    };
    
    console.log('Credentials:', {
        apiKey: credentials.apiKey,
        apiSecret: credentials.apiSecret ? '***' + credentials.apiSecret.slice(-4) : 'undefined',
        testnet: credentials.testnet
    });
    
    const deltaService = new DeltaExchangeService(credentials);
    
    try {
        console.log('\n🔄 Waiting for service to be ready...');

        // Wait for service to be ready
        let attempts = 0;
        while (!deltaService.isReady() && attempts < 30) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
        }

        if (!deltaService.isReady()) {
            throw new Error('Service not ready after 30 seconds');
        }

        console.log('✅ Service is ready');

        console.log('\n💰 Testing balance fetch...');
        const balances = await deltaService.getBalances();
        console.log('✅ Balances fetched:', balances.length);
        console.log('Balance data:', JSON.stringify(balances, null, 2));
        
        if (balances.length > 0) {
            console.log('\n🎉 SUCCESS! Found balances in your account:');
            balances.forEach(balance => {
                console.log(`  ${balance.asset?.symbol || 'Unknown'}: ${balance.balance}`);
            });
        } else {
            console.log('\n⚠️ No balances found, but API is working correctly');
        }
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.error('Stack:', error.stack);
    }
}

testDeltaService();
