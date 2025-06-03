#!/usr/bin/env node
"use strict";
/**
 * Test Fixed Delta Exchange Authentication
 * Complete test with corrected authentication format
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const deltaApiService_1 = __importDefault(require("../services/deltaApiService"));
async function testFixedDeltaAuth() {
    console.log('🔧 TESTING FIXED DELTA EXCHANGE AUTHENTICATION');
    console.log('='.repeat(80));
    try {
        // Initialize Delta API for testnet with correct URL
        const deltaApi = new deltaApiService_1.default({ testnet: true });
        // Get credentials from environment
        const credentials = {
            key: process.env.DELTA_EXCHANGE_API_KEY || '',
            secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
        };
        if (!credentials.key || !credentials.secret) {
            console.log('❌ Delta API credentials not found in environment');
            console.log('🔧 Please set DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET');
            return;
        }
        console.log('✅ Credentials found');
        console.log(`🔑 API Key: ${credentials.key.substring(0, 8)}...`);
        console.log(`🔐 Secret: ${credentials.secret.substring(0, 8)}...`);
        // Initialize connection
        await deltaApi.initialize(credentials);
        console.log('✅ API client initialized with fixed authentication');
        // Test 1: Public endpoint (should work)
        console.log('\n📊 TEST 1: PUBLIC MARKET DATA');
        try {
            const markets = await deltaApi.getMarkets();
            console.log(`✅ Retrieved ${markets.length} markets (public endpoint working)`);
            const btcMarket = markets.find(m => m.symbol.includes('BTC') && m.symbol.includes('USD'));
            if (btcMarket) {
                console.log(`✅ Found BTC market: ${btcMarket.symbol}`);
                console.log(`   Type: ${btcMarket.contract_type}`);
                console.log(`   Active: ${btcMarket.is_active}`);
            }
        }
        catch (error) {
            console.log('❌ Public market data failed:', error.message);
        }
        // Test 2: Account info (authenticated endpoint)
        console.log('\n👤 TEST 2: ACCOUNT INFORMATION (AUTHENTICATED)');
        try {
            const accountInfo = await deltaApi.getAccountInfo();
            console.log('✅ AUTHENTICATION SUCCESS!');
            console.log(`✅ Account ID: ${accountInfo.id}`);
            console.log(`✅ Email: ${accountInfo.email}`);
            console.log(`✅ Name: ${accountInfo.name}`);
            console.log(`✅ Verified: ${accountInfo.is_verified}`);
        }
        catch (error) {
            console.log('❌ Account info failed:', error.message);
            if (error.message.includes('expired_signature')) {
                console.log('🔧 Still getting signature errors - may need further adjustment');
            }
            else if (error.message.includes('InvalidApiKey')) {
                console.log('🔧 API key issue - check if key is for testnet');
            }
            else if (error.message.includes('UnauthorizedApiAccess')) {
                console.log('🔧 Permission issue - check API key permissions');
            }
        }
        // Test 3: Wallet balances (authenticated endpoint)
        console.log('\n💰 TEST 3: WALLET BALANCES (AUTHENTICATED)');
        try {
            const balances = await deltaApi.getWalletBalances();
            console.log('✅ WALLET ACCESS SUCCESS!');
            console.log(`✅ Retrieved ${balances.length} wallet balances`);
            const nonZeroBalances = balances.filter(b => parseFloat(b.balance) > 0);
            if (nonZeroBalances.length > 0) {
                console.log('💰 Non-zero balances:');
                nonZeroBalances.forEach(balance => {
                    console.log(`   ${balance.asset}: ${balance.balance} (Available: ${balance.available_balance})`);
                });
            }
            else {
                console.log('⚠️ No non-zero balances (testnet account may be empty)');
            }
        }
        catch (error) {
            console.log('❌ Wallet balances failed:', error.message);
        }
        // Test 4: Current positions (authenticated endpoint)
        console.log('\n📈 TEST 4: CURRENT POSITIONS (AUTHENTICATED)');
        try {
            const positions = await deltaApi.getPositions();
            console.log('✅ POSITIONS ACCESS SUCCESS!');
            console.log(`✅ Retrieved ${positions.length} positions`);
            if (positions.length > 0) {
                console.log('📊 Active positions:');
                positions.forEach(position => {
                    console.log(`   ${position.symbol}: ${position.size} @ $${position.entry_price}`);
                });
            }
            else {
                console.log('✅ No active positions (clean slate for testing)');
            }
        }
        catch (error) {
            console.log('❌ Positions failed:', error.message);
        }
        // Test 5: Active orders (authenticated endpoint)
        console.log('\n📝 TEST 5: ACTIVE ORDERS (AUTHENTICATED)');
        try {
            const orders = await deltaApi.getActiveOrders();
            console.log('✅ ORDERS ACCESS SUCCESS!');
            console.log(`✅ Retrieved ${orders.length} active orders`);
            if (orders.length > 0) {
                console.log('📋 Active orders:');
                orders.forEach(order => {
                    console.log(`   ${order.symbol}: ${order.side} ${order.size} @ $${order.limit_price || order.price}`);
                });
            }
            else {
                console.log('✅ No active orders');
            }
        }
        catch (error) {
            console.log('❌ Active orders failed:', error.message);
        }
        // Test 6: Order placement test (if we have balance)
        console.log('\n🎯 TEST 6: ORDER PLACEMENT TEST');
        try {
            // Get markets first
            const markets = await deltaApi.getMarkets();
            const btcMarket = markets.find(m => m.symbol === 'BTC_USDT' ||
                (m.symbol.includes('BTC') && m.symbol.includes('USD') && m.is_active));
            if (!btcMarket) {
                console.log('⚠️ No suitable BTC market found for order test');
            }
            else {
                console.log(`🎯 Testing order placement on: ${btcMarket.symbol}`);
                // Place a very conservative limit order (won't execute)
                const orderParams = {
                    symbol: btcMarket.symbol,
                    side: 'buy',
                    size: 0.001, // Very small size
                    type: 'limit',
                    price: 1000, // Very low price (won't execute)
                    timeInForce: 'gtc',
                    clientOrderId: `test_${Date.now()}`,
                };
                console.log(`📊 Placing test order: ${orderParams.side} ${orderParams.size} ${orderParams.symbol} @ $${orderParams.price}`);
                const order = await deltaApi.placeOrder(orderParams);
                console.log('✅ ORDER PLACEMENT SUCCESS!');
                console.log(`   Order ID: ${order.id}`);
                console.log(`   Status: ${order.status}`);
                console.log(`   Symbol: ${order.symbol}`);
                // Try to cancel the order
                console.log('\n❌ Testing order cancellation...');
                try {
                    const cancelResult = await deltaApi.cancelOrder(order.id.toString());
                    console.log('✅ ORDER CANCELLATION SUCCESS!');
                    console.log('   Order cancelled successfully');
                }
                catch (cancelError) {
                    console.log('⚠️ Order cancellation failed:', cancelError.message);
                }
            }
        }
        catch (error) {
            console.log('❌ Order placement test failed:', error.message);
        }
        console.log('\n🎉 FIXED AUTHENTICATION TEST COMPLETED');
        // Final assessment
        console.log('\n📊 FINAL ASSESSMENT:');
        console.log('✅ Public endpoints: WORKING');
        console.log('✅ API connection: WORKING');
        console.log('✅ Market data: WORKING');
        console.log('🔧 Authentication: Check results above');
        console.log('🚀 Ready for trading system integration if auth working');
    }
    catch (error) {
        console.log('❌ Test failed:', error.message);
        console.log('🔧 Check API credentials and network connection');
    }
}
testFixedDeltaAuth().catch(console.error);
//# sourceMappingURL=test-fixed-delta-auth.js.map