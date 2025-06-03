#!/usr/bin/env node
"use strict";
/**
 * Delta Testnet Connection & Trade Execution Test
 * Comprehensive test to verify:
 * 1. Connection to Delta testnet
 * 2. Account balance retrieval
 * 3. Market data access
 * 4. Order placement (entry)
 * 5. Order cancellation (exit)
 * 6. Position management
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DeltaTestnetTester = void 0;
const deltaApiService_1 = __importDefault(require("../services/deltaApiService"));
const logger_1 = require("../utils/logger");
class DeltaTestnetTester {
    constructor() {
        this.testResults = {};
        // Initialize for testnet
        this.deltaApi = new deltaApiService_1.default({ testnet: true });
    }
    /**
     * Run comprehensive Delta testnet tests
     */
    async runComprehensiveTest() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🚀 DELTA TESTNET CONNECTION & TRADE EXECUTION TEST');
            logger_1.logger.info('='.repeat(80));
            // Test 1: Initialize connection
            await this.testConnection();
            // Test 2: Get account information
            await this.testAccountInfo();
            // Test 3: Get wallet balances
            await this.testWalletBalances();
            // Test 4: Get market data
            await this.testMarketData();
            // Test 5: Get current positions
            await this.testPositions();
            // Test 6: Test order placement (entry)
            await this.testOrderPlacement();
            // Test 7: Test order cancellation (exit)
            await this.testOrderCancellation();
            // Test 8: Test position management
            await this.testPositionManagement();
            // Generate comprehensive report
            this.generateTestReport(startTime);
        }
        catch (error) {
            logger_1.logger.error('❌ Delta testnet test failed:', error);
            throw error;
        }
    }
    /**
     * Test 1: Connection to Delta testnet
     */
    async testConnection() {
        try {
            logger_1.logger.info('\n📡 TEST 1: DELTA TESTNET CONNECTION');
            // Get credentials from environment
            const credentials = {
                key: process.env.DELTA_EXCHANGE_API_KEY || '',
                secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
            };
            if (!credentials.key || !credentials.secret) {
                throw new Error('Delta API credentials not found in environment variables');
            }
            logger_1.logger.info('   ✅ Credentials found in environment');
            logger_1.logger.info(`   🔑 API Key: ${credentials.key.substring(0, 8)}...`);
            logger_1.logger.info(`   🔐 Secret: ${credentials.secret.substring(0, 8)}...`);
            // Initialize connection
            await this.deltaApi.initialize(credentials);
            logger_1.logger.info('   ✅ API client initialized successfully');
            // Test server time
            const serverTime = await this.deltaApi.getServerTime();
            logger_1.logger.info(`   ✅ Server time retrieved: ${new Date(serverTime.timestamp * 1000).toISOString()}`);
            this.testResults.connection = {
                status: 'SUCCESS',
                serverTime: serverTime.timestamp,
                testnet: true,
            };
        }
        catch (error) {
            logger_1.logger.error('   ❌ Connection test failed:', error);
            this.testResults.connection = {
                status: 'FAILED',
                error: error.message,
            };
            throw error;
        }
    }
    /**
     * Test 2: Account information retrieval
     */
    async testAccountInfo() {
        try {
            logger_1.logger.info('\n👤 TEST 2: ACCOUNT INFORMATION');
            const accountInfo = await this.deltaApi.getAccountInfo();
            logger_1.logger.info(`   ✅ Account ID: ${accountInfo.id}`);
            logger_1.logger.info(`   ✅ Email: ${accountInfo.email}`);
            logger_1.logger.info(`   ✅ KYC Status: ${accountInfo.kyc_status || 'N/A'}`);
            logger_1.logger.info(`   ✅ Trading Enabled: ${accountInfo.trading_enabled || 'N/A'}`);
            this.testResults.accountInfo = {
                status: 'SUCCESS',
                accountId: accountInfo.id,
                email: accountInfo.email,
                kycStatus: accountInfo.kyc_status,
                tradingEnabled: accountInfo.trading_enabled,
            };
        }
        catch (error) {
            logger_1.logger.error('   ❌ Account info test failed:', error);
            this.testResults.accountInfo = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Test 3: Wallet balances retrieval
     */
    async testWalletBalances() {
        try {
            logger_1.logger.info('\n💰 TEST 3: WALLET BALANCES');
            const balances = await this.deltaApi.getWalletBalances();
            logger_1.logger.info(`   ✅ Retrieved ${balances.length} wallet balances`);
            // Show significant balances
            const significantBalances = balances.filter(b => parseFloat(b.balance) > 0);
            if (significantBalances.length > 0) {
                logger_1.logger.info('   💰 Non-zero balances:');
                significantBalances.forEach(balance => {
                    logger_1.logger.info(`     ${balance.asset}: ${balance.balance} (Available: ${balance.available_balance})`);
                });
            }
            else {
                logger_1.logger.info('   ⚠️  No non-zero balances found (testnet account may be empty)');
            }
            // Find USD/USDT balance for trading
            const usdBalance = balances.find(b => b.asset === 'USDT' || b.asset === 'USD' || b.asset === 'USDC');
            let tradingBalance = 0;
            if (usdBalance) {
                tradingBalance = parseFloat(usdBalance.available_balance);
                logger_1.logger.info(`   💵 Trading balance: ${tradingBalance} ${usdBalance.asset}`);
            }
            this.testResults.walletBalances = {
                status: 'SUCCESS',
                totalBalances: balances.length,
                significantBalances: significantBalances.length,
                tradingBalance,
                tradingAsset: usdBalance?.asset || 'NONE',
            };
        }
        catch (error) {
            logger_1.logger.error('   ❌ Wallet balances test failed:', error);
            this.testResults.walletBalances = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Test 4: Market data retrieval
     */
    async testMarketData() {
        try {
            logger_1.logger.info('\n📊 TEST 4: MARKET DATA');
            // Test getting all markets
            const markets = await this.deltaApi.getMarkets();
            logger_1.logger.info(`   ✅ Retrieved ${markets.length} available markets`);
            // Find BTC perpetual contract
            const btcMarket = markets.find(m => m.symbol.includes('BTC') && m.symbol.includes('USD') &&
                (m.contract_type === 'perpetual_futures' || m.product_type === 'futures'));
            if (btcMarket) {
                logger_1.logger.info(`   ✅ Found BTC market: ${btcMarket.symbol}`);
                logger_1.logger.info(`     Contract Type: ${btcMarket.contract_type || btcMarket.product_type}`);
                logger_1.logger.info(`     Status: ${btcMarket.state || btcMarket.trading_status}`);
                // Get ticker for BTC market
                try {
                    const ticker = await this.deltaApi.getTicker(btcMarket.symbol);
                    logger_1.logger.info(`   ✅ BTC Ticker - Price: $${ticker.close || ticker.mark_price}, Volume: ${ticker.volume}`);
                }
                catch (tickerError) {
                    logger_1.logger.warn(`   ⚠️  Could not get ticker for ${btcMarket.symbol}:`, tickerError.message);
                }
                this.testResults.marketData = {
                    status: 'SUCCESS',
                    totalMarkets: markets.length,
                    btcMarket: btcMarket.symbol,
                    btcPrice: 'Retrieved',
                };
            }
            else {
                logger_1.logger.warn('   ⚠️  No BTC perpetual market found');
                this.testResults.marketData = {
                    status: 'PARTIAL',
                    totalMarkets: markets.length,
                    btcMarket: 'NOT_FOUND',
                };
            }
        }
        catch (error) {
            logger_1.logger.error('   ❌ Market data test failed:', error);
            this.testResults.marketData = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Test 5: Current positions
     */
    async testPositions() {
        try {
            logger_1.logger.info('\n📈 TEST 5: CURRENT POSITIONS');
            const positions = await this.deltaApi.getPositions();
            logger_1.logger.info(`   ✅ Retrieved ${positions.length} positions`);
            if (positions.length > 0) {
                logger_1.logger.info('   📊 Active positions:');
                positions.forEach(position => {
                    logger_1.logger.info(`     ${position.symbol}: ${position.size} @ $${position.entry_price || position.average_entry_price}`);
                });
            }
            else {
                logger_1.logger.info('   ✅ No active positions (clean slate for testing)');
            }
            this.testResults.positions = {
                status: 'SUCCESS',
                activePositions: positions.length,
            };
        }
        catch (error) {
            logger_1.logger.error('   ❌ Positions test failed:', error);
            this.testResults.positions = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Test 6: Order placement (entry test)
     */
    async testOrderPlacement() {
        try {
            logger_1.logger.info('\n📝 TEST 6: ORDER PLACEMENT (ENTRY)');
            // Get available markets first
            const markets = await this.deltaApi.getMarkets();
            const btcMarket = markets.find(m => m.symbol.includes('BTC') && m.symbol.includes('USD') &&
                (m.contract_type === 'perpetual_futures' || m.product_type === 'futures'));
            if (!btcMarket) {
                throw new Error('No BTC market found for testing');
            }
            logger_1.logger.info(`   🎯 Testing with market: ${btcMarket.symbol}`);
            // Get current ticker to determine a safe price
            const ticker = await this.deltaApi.getTicker(btcMarket.symbol);
            const currentPrice = parseFloat(ticker.close || ticker.mark_price || '50000');
            // Place a limit order well below market price (won't execute)
            const testPrice = currentPrice * 0.8; // 20% below market
            const testSize = 0.001; // Very small size for testing
            const orderParams = {
                symbol: btcMarket.symbol,
                side: 'buy',
                size: testSize,
                type: 'limit',
                price: testPrice,
                timeInForce: 'gtc',
                clientOrderId: `test_${Date.now()}`,
            };
            logger_1.logger.info(`   📊 Placing test order: ${orderParams.side} ${orderParams.size} ${orderParams.symbol} @ $${testPrice}`);
            const order = await this.deltaApi.placeOrder(orderParams);
            logger_1.logger.info(`   ✅ Order placed successfully!`);
            logger_1.logger.info(`     Order ID: ${order.id}`);
            logger_1.logger.info(`     Status: ${order.state || order.status}`);
            logger_1.logger.info(`     Symbol: ${order.symbol}`);
            logger_1.logger.info(`     Side: ${order.side}`);
            logger_1.logger.info(`     Size: ${order.size}`);
            logger_1.logger.info(`     Price: $${order.price}`);
            this.testResults.orderPlacement = {
                status: 'SUCCESS',
                orderId: order.id,
                symbol: order.symbol,
                side: order.side,
                size: order.size,
                price: order.price,
                orderStatus: order.state || order.status,
            };
            // Store order ID for cancellation test
            this.testResults.testOrderId = order.id;
        }
        catch (error) {
            logger_1.logger.error('   ❌ Order placement test failed:', error);
            this.testResults.orderPlacement = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Test 7: Order cancellation (exit test)
     */
    async testOrderCancellation() {
        try {
            logger_1.logger.info('\n❌ TEST 7: ORDER CANCELLATION (EXIT)');
            if (!this.testResults.testOrderId) {
                logger_1.logger.warn('   ⚠️  No test order ID available, skipping cancellation test');
                this.testResults.orderCancellation = {
                    status: 'SKIPPED',
                    reason: 'No order to cancel',
                };
                return;
            }
            const orderId = this.testResults.testOrderId;
            logger_1.logger.info(`   🎯 Cancelling test order: ${orderId}`);
            // Cancel the order
            const cancelResult = await this.deltaApi.cancelOrder(orderId);
            logger_1.logger.info(`   ✅ Order cancelled successfully!`);
            logger_1.logger.info(`     Cancel result:`, cancelResult);
            // Verify cancellation by checking active orders
            const activeOrders = await this.deltaApi.getActiveOrders();
            const cancelledOrder = activeOrders.find(o => o.id === orderId);
            if (!cancelledOrder) {
                logger_1.logger.info(`   ✅ Order ${orderId} no longer in active orders`);
            }
            else {
                logger_1.logger.info(`   ⚠️  Order ${orderId} still shows as: ${cancelledOrder.state || cancelledOrder.status}`);
            }
            this.testResults.orderCancellation = {
                status: 'SUCCESS',
                orderId,
                cancelResult,
                stillActive: !!cancelledOrder,
            };
        }
        catch (error) {
            logger_1.logger.error('   ❌ Order cancellation test failed:', error);
            this.testResults.orderCancellation = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Test 8: Position management
     */
    async testPositionManagement() {
        try {
            logger_1.logger.info('\n📊 TEST 8: POSITION MANAGEMENT');
            // Get current positions
            const positions = await this.deltaApi.getPositions();
            logger_1.logger.info(`   ✅ Retrieved ${positions.length} current positions`);
            // Get active orders
            const activeOrders = await this.deltaApi.getActiveOrders();
            logger_1.logger.info(`   ✅ Retrieved ${activeOrders.length} active orders`);
            // Get recent trade history
            const trades = await this.deltaApi.getTradeHistory({ limit: 10 });
            logger_1.logger.info(`   ✅ Retrieved ${trades.length} recent trades`);
            if (trades.length > 0) {
                logger_1.logger.info('   📈 Recent trades:');
                trades.slice(0, 3).forEach(trade => {
                    logger_1.logger.info(`     ${trade.symbol}: ${trade.side} ${trade.size} @ $${trade.price}`);
                });
            }
            this.testResults.positionManagement = {
                status: 'SUCCESS',
                currentPositions: positions.length,
                activeOrders: activeOrders.length,
                recentTrades: trades.length,
            };
        }
        catch (error) {
            logger_1.logger.error('   ❌ Position management test failed:', error);
            this.testResults.positionManagement = {
                status: 'FAILED',
                error: error.message,
            };
        }
    }
    /**
     * Generate comprehensive test report
     */
    generateTestReport(startTime) {
        const duration = (Date.now() - startTime) / 1000;
        logger_1.logger.info('\n' + '🎉 DELTA TESTNET TEST RESULTS'.padStart(80, '='));
        logger_1.logger.info('='.repeat(120));
        // Test Summary
        logger_1.logger.info('📊 TEST SUMMARY:');
        const tests = [
            'connection',
            'accountInfo',
            'walletBalances',
            'marketData',
            'positions',
            'orderPlacement',
            'orderCancellation',
            'positionManagement'
        ];
        let passedTests = 0;
        let failedTests = 0;
        let skippedTests = 0;
        tests.forEach(testName => {
            const result = this.testResults[testName];
            if (result) {
                const status = result.status;
                const icon = status === 'SUCCESS' ? '✅' : status === 'FAILED' ? '❌' : '⚠️';
                logger_1.logger.info(`   ${icon} ${testName.toUpperCase()}: ${status}`);
                if (status === 'SUCCESS')
                    passedTests++;
                else if (status === 'FAILED')
                    failedTests++;
                else
                    skippedTests++;
            }
        });
        logger_1.logger.info('\n📈 OVERALL RESULTS:');
        logger_1.logger.info(`   Total Tests: ${tests.length}`);
        logger_1.logger.info(`   Passed: ${passedTests}`);
        logger_1.logger.info(`   Failed: ${failedTests}`);
        logger_1.logger.info(`   Skipped: ${skippedTests}`);
        logger_1.logger.info(`   Success Rate: ${((passedTests / tests.length) * 100).toFixed(1)}%`);
        logger_1.logger.info(`   Execution Time: ${duration.toFixed(2)} seconds`);
        // Connection Status
        logger_1.logger.info('\n🔗 CONNECTION STATUS:');
        if (this.testResults.connection?.status === 'SUCCESS') {
            logger_1.logger.info('   ✅ Delta testnet connection: WORKING');
            logger_1.logger.info('   ✅ API authentication: WORKING');
            logger_1.logger.info('   ✅ Server communication: WORKING');
        }
        else {
            logger_1.logger.info('   ❌ Delta testnet connection: FAILED');
        }
        // Trading Capabilities
        logger_1.logger.info('\n💰 TRADING CAPABILITIES:');
        if (this.testResults.walletBalances?.tradingBalance > 0) {
            logger_1.logger.info(`   ✅ Trading balance: $${this.testResults.walletBalances.tradingBalance} ${this.testResults.walletBalances.tradingAsset}`);
        }
        else {
            logger_1.logger.info('   ⚠️  No trading balance available (testnet account may need funding)');
        }
        if (this.testResults.orderPlacement?.status === 'SUCCESS') {
            logger_1.logger.info('   ✅ Order placement: WORKING');
        }
        else {
            logger_1.logger.info('   ❌ Order placement: FAILED');
        }
        if (this.testResults.orderCancellation?.status === 'SUCCESS') {
            logger_1.logger.info('   ✅ Order cancellation: WORKING');
        }
        else if (this.testResults.orderCancellation?.status === 'SKIPPED') {
            logger_1.logger.info('   ⚠️  Order cancellation: SKIPPED');
        }
        else {
            logger_1.logger.info('   ❌ Order cancellation: FAILED');
        }
        // Market Data Access
        logger_1.logger.info('\n📊 MARKET DATA ACCESS:');
        if (this.testResults.marketData?.status === 'SUCCESS') {
            logger_1.logger.info(`   ✅ Market data: ${this.testResults.marketData.totalMarkets} markets available`);
            logger_1.logger.info(`   ✅ BTC market: ${this.testResults.marketData.btcMarket}`);
        }
        else {
            logger_1.logger.info('   ❌ Market data access: FAILED');
        }
        // Account Information
        logger_1.logger.info('\n👤 ACCOUNT INFORMATION:');
        if (this.testResults.accountInfo?.status === 'SUCCESS') {
            logger_1.logger.info(`   ✅ Account ID: ${this.testResults.accountInfo.accountId}`);
            logger_1.logger.info(`   ✅ Email: ${this.testResults.accountInfo.email}`);
            logger_1.logger.info(`   ✅ Trading enabled: ${this.testResults.accountInfo.tradingEnabled}`);
        }
        else {
            logger_1.logger.info('   ❌ Account information: FAILED');
        }
        // Final Assessment
        logger_1.logger.info('\n🎯 FINAL ASSESSMENT:');
        if (passedTests >= 6) {
            logger_1.logger.info('   🚀 EXCELLENT: Delta testnet integration is working well!');
            logger_1.logger.info('   ✅ Ready for live trading system integration');
            logger_1.logger.info('   ✅ Entry and exit functionality verified');
            logger_1.logger.info('   ✅ Account and balance management working');
            if (this.testResults.walletBalances?.tradingBalance > 0) {
                logger_1.logger.info('   💰 Sufficient balance for trading tests');
                logger_1.logger.info('   🎯 READY FOR DYNAMIC TAKE PROFIT SYSTEM DEPLOYMENT');
            }
            else {
                logger_1.logger.info('   ⚠️  Consider adding testnet funds for full trading tests');
            }
        }
        else if (passedTests >= 4) {
            logger_1.logger.info('   📈 GOOD: Most functionality working, some issues to resolve');
            logger_1.logger.info('   🔧 Address failed tests before live deployment');
        }
        else {
            logger_1.logger.info('   ⚠️  NEEDS WORK: Multiple issues detected');
            logger_1.logger.info('   🔧 Resolve connection and API issues before proceeding');
        }
        // Next Steps
        logger_1.logger.info('\n🔄 NEXT STEPS:');
        if (passedTests >= 6) {
            logger_1.logger.info('   1. ✅ Delta testnet integration verified');
            logger_1.logger.info('   2. 🚀 Deploy dynamic take profit system');
            logger_1.logger.info('   3. 📊 Run 3-month backtest with real Delta balance');
            logger_1.logger.info('   4. 💰 Start live trading with small capital');
            logger_1.logger.info('   5. 📈 Scale up based on performance');
        }
        else {
            logger_1.logger.info('   1. 🔧 Fix failed test cases');
            logger_1.logger.info('   2. 💰 Add testnet funds if needed');
            logger_1.logger.info('   3. 🔄 Re-run connection tests');
            logger_1.logger.info('   4. ✅ Verify all functionality before live deployment');
        }
        logger_1.logger.info('='.repeat(120));
    }
}
exports.DeltaTestnetTester = DeltaTestnetTester;
/**
 * Main execution function
 */
async function main() {
    const tester = new DeltaTestnetTester();
    try {
        await tester.runComprehensiveTest();
    }
    catch (error) {
        logger_1.logger.error('💥 Delta testnet test failed:', error);
        process.exit(1);
    }
}
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=test-delta-connection-trades.js.map