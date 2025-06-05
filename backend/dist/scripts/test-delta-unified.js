"use strict";
/**
 * Test Delta Exchange Unified Service
 * Quick test to verify the new Delta Exchange integration works
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.testDeltaExchangeUnified = testDeltaExchangeUnified;
const DeltaExchangeUnified_1 = require("../services/DeltaExchangeUnified");
const DeltaBotManager_1 = require("../services/DeltaBotManager");
async function testDeltaExchangeUnified() {
    console.log('🚀 Testing Delta Exchange Unified Service...\n');
    try {
        // Test 1: Initialize Delta Exchange Service
        console.log('1. 🔧 Initializing Delta Exchange Service...');
        const credentials = {
            apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
            apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
            testnet: true
        };
        if (!credentials.apiKey || !credentials.apiSecret) {
            throw new Error('Delta Exchange API credentials not configured');
        }
        const deltaService = new DeltaExchangeUnified_1.DeltaExchangeUnified(credentials);
        // Wait for initialization
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Initialization timeout')), 30000);
            deltaService.once('initialized', () => {
                clearTimeout(timeout);
                resolve(true);
            });
            deltaService.once('error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
        console.log('✅ Delta Exchange Service initialized successfully\n');
        // Test 2: Check Products
        console.log('2. 📦 Testing product loading...');
        const btcProduct = deltaService.getProductBySymbol('BTCUSD');
        const ethProduct = deltaService.getProductBySymbol('ETHUSD');
        if (btcProduct) {
            console.log(`✅ BTC/USD Product found: ID ${btcProduct.id}, State: ${btcProduct.state}`);
        }
        else {
            console.log('❌ BTC/USD Product not found');
        }
        if (ethProduct) {
            console.log(`✅ ETH/USD Product found: ID ${ethProduct.id}, State: ${ethProduct.state}`);
        }
        else {
            console.log('❌ ETH/USD Product not found');
        }
        const allProducts = deltaService.getAllProducts();
        console.log(`📊 Total products loaded: ${allProducts.length}`);
        const perpetualProducts = deltaService.getPerpetualProducts();
        console.log(`🔄 Perpetual products: ${perpetualProducts.length}\n`);
        // Test 3: Test Authentication
        console.log('3. 🔐 Testing authentication...');
        try {
            const balance = await deltaService.getBalance();
            console.log(`✅ Authentication successful - Balance entries: ${balance.length}`);
            // Show first balance entry if available
            if (balance.length > 0) {
                const firstBalance = balance[0];
                console.log(`💰 Sample balance: ${firstBalance.asset_symbol} - ${firstBalance.available_balance}`);
            }
        }
        catch (error) {
            console.log(`❌ Authentication failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        console.log('');
        // Test 4: Test Market Data
        console.log('4. 📈 Testing market data...');
        try {
            if (btcProduct) {
                const marketData = await deltaService.getMarketData('BTCUSD');
                console.log(`✅ BTC/USD market data retrieved`);
                console.log(`📊 Mark Price: ${marketData.mark_price || 'N/A'}`);
                console.log(`📊 Last Price: ${marketData.last_price || 'N/A'}`);
            }
        }
        catch (error) {
            console.log(`❌ Market data failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        console.log('');
        // Test 5: Test Bot Manager
        console.log('5. 🤖 Testing Bot Manager...');
        try {
            const botManager = new DeltaBotManager_1.DeltaBotManager();
            await botManager.initialize();
            console.log('✅ Bot Manager initialized successfully');
            // Test bot creation
            const testBotConfig = {
                id: 'test-bot-001',
                name: 'Test Delta Bot',
                symbol: 'BTCUSD',
                strategy: 'momentum',
                capital: 100,
                leverage: 3,
                riskPerTrade: 2,
                maxPositions: 1,
                stopLoss: 5,
                takeProfit: 10,
                enabled: true,
                testnet: true
            };
            const botId = await botManager.createBot(testBotConfig);
            console.log(`✅ Test bot created: ${botId}`);
            // Get bot status
            const status = botManager.getBotStatus(botId);
            console.log(`📊 Bot status: ${status.status}`);
            // Clean up
            await botManager.removeBot(botId);
            console.log('🧹 Test bot removed');
            await botManager.cleanup();
            console.log('✅ Bot Manager cleaned up');
        }
        catch (error) {
            console.log(`❌ Bot Manager test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        console.log('');
        // Test 6: WebSocket Connection (brief test)
        console.log('6. 🔗 Testing WebSocket connection...');
        try {
            deltaService.connectWebSocket(['BTCUSD']);
            // Wait a moment for connection
            await new Promise(resolve => setTimeout(resolve, 3000));
            deltaService.disconnectWebSocket();
            console.log('✅ WebSocket connection test completed');
        }
        catch (error) {
            console.log(`❌ WebSocket test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        console.log('');
        // Cleanup
        deltaService.cleanup();
        console.log('🧹 Delta Exchange Service cleaned up');
        console.log('🎉 All tests completed successfully!');
    }
    catch (error) {
        console.error('❌ Test failed:', error instanceof Error ? error.message : 'Unknown error');
        process.exit(1);
    }
}
// Run the test
if (require.main === module) {
    // Load environment variables
    require('dotenv').config({ path: require('path').resolve(__dirname, '../../../.env') });
    testDeltaExchangeUnified()
        .then(() => {
        console.log('\n✅ Test completed successfully');
        process.exit(0);
    })
        .catch((error) => {
        console.error('\n❌ Test failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=test-delta-unified.js.map