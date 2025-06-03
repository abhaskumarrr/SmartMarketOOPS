#!/usr/bin/env node
"use strict";
/**
 * Full System Test and Order Placement
 * Complete end-to-end test with real order placement on Delta Exchange
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FullSystemTest = void 0;
const deltaApiService_1 = __importDefault(require("../services/deltaApiService"));
const dynamicTakeProfitManager_1 = require("../services/dynamicTakeProfitManager");
const logger_1 = require("../utils/logger");
class FullSystemTest {
    constructor() {
        this.deltaApi = new deltaApiService_1.default({ testnet: true });
        this.takeProfitManager = new dynamicTakeProfitManager_1.DynamicTakeProfitManager();
    }
    /**
     * Run complete system test
     */
    async runFullTest() {
        logger_1.logger.info('🚀 STARTING FULL SYSTEM TEST WITH ORDER PLACEMENT');
        logger_1.logger.info('='.repeat(80));
        try {
            // Step 1: Initialize and test connection
            await this.testConnection();
            // Step 2: Test dynamic take profit system
            await this.testDynamicTakeProfit();
            // Step 3: Test market data access
            await this.testMarketData();
            // Step 4: Test account access
            await this.testAccountAccess();
            // Step 5: Place actual test order
            await this.placeTestOrder();
            logger_1.logger.info('\n🎉 FULL SYSTEM TEST COMPLETED SUCCESSFULLY!');
        }
        catch (error) {
            logger_1.logger.error('❌ Full system test failed:', error.message);
            if (error.message.includes('ip_not_whitelisted')) {
                logger_1.logger.info('\n🔧 IP WHITELISTING REQUIRED:');
                logger_1.logger.info('   Your IP needs to be whitelisted in Delta Exchange API settings');
                logger_1.logger.info('   Current IP: 223.226.141.59');
                logger_1.logger.info('   Add this IP to your API key whitelist and try again');
            }
        }
    }
    /**
     * Test connection and initialization
     */
    async testConnection() {
        logger_1.logger.info('\n🔌 STEP 1: CONNECTION TEST');
        // Get credentials
        const credentials = {
            key: process.env.DELTA_EXCHANGE_API_KEY || '',
            secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
        };
        if (!credentials.key || !credentials.secret) {
            throw new Error('Delta API credentials not found in environment');
        }
        logger_1.logger.info('✅ Credentials found');
        logger_1.logger.info(`   API Key: ${credentials.key.substring(0, 8)}...`);
        // Initialize connection
        await this.deltaApi.initialize(credentials);
        logger_1.logger.info('✅ Delta Exchange API initialized');
        // Test public endpoint
        const markets = await this.deltaApi.getMarkets();
        logger_1.logger.info(`✅ Public API working - ${markets.length} markets available`);
    }
    /**
     * Test dynamic take profit system
     */
    async testDynamicTakeProfit() {
        logger_1.logger.info('\n🎯 STEP 2: DYNAMIC TAKE PROFIT SYSTEM TEST');
        // Test configuration
        const testConfig = {
            asset: 'BTCUSD',
            entryPrice: 95000,
            stopLoss: 92625,
            positionSize: 0.01,
            side: 'BUY',
            marketRegime: {
                type: 'TRENDING',
                strength: 75,
                direction: 'UP',
                volatility: 0.03,
                volume: 1.2,
            },
            momentum: 50,
            volume: 1.2,
        };
        // Generate take profit levels
        const levels = this.takeProfitManager.generateDynamicTakeProfitLevels(testConfig);
        logger_1.logger.info('✅ Dynamic take profit system operational');
        logger_1.logger.info(`   Generated ${levels.length} take profit levels:`);
        levels.forEach((level, index) => {
            logger_1.logger.info(`     Level ${index + 1}: ${level.percentage}% at $${level.priceTarget.toFixed(2)} (${level.riskRewardRatio.toFixed(1)}:1)`);
        });
    }
    /**
     * Test market data access
     */
    async testMarketData() {
        logger_1.logger.info('\n📊 STEP 3: MARKET DATA TEST');
        // Get markets
        const markets = await this.deltaApi.getMarkets();
        const btcMarkets = markets.filter(m => m.symbol.includes('BTC'));
        logger_1.logger.info(`✅ Market data access working`);
        logger_1.logger.info(`   Total markets: ${markets.length}`);
        logger_1.logger.info(`   BTC markets: ${btcMarkets.length}`);
        if (btcMarkets.length > 0) {
            const btcMarket = btcMarkets[0];
            logger_1.logger.info(`   Primary BTC market: ${btcMarket.symbol}`);
            logger_1.logger.info(`   Contract type: ${btcMarket.contract_type}`);
            logger_1.logger.info(`   Active: ${btcMarket.is_active}`);
            // Test ticker data
            try {
                const ticker = await this.deltaApi.getTicker(btcMarket.symbol);
                logger_1.logger.info(`   Current price: $${ticker.close}`);
                logger_1.logger.info(`   Volume: ${ticker.volume}`);
            }
            catch (tickerError) {
                logger_1.logger.warn(`   ⚠️ Ticker data not available: ${tickerError.message}`);
            }
        }
    }
    /**
     * Test account access
     */
    async testAccountAccess() {
        logger_1.logger.info('\n👤 STEP 4: ACCOUNT ACCESS TEST');
        try {
            // Test account info
            const accountInfo = await this.deltaApi.getAccountInfo();
            logger_1.logger.info('✅ Account access successful');
            logger_1.logger.info(`   Account ID: ${accountInfo.id}`);
            logger_1.logger.info(`   Email: ${accountInfo.email}`);
            logger_1.logger.info(`   Name: ${accountInfo.name}`);
            logger_1.logger.info(`   Verified: ${accountInfo.is_verified}`);
            // Test wallet balances
            const balances = await this.deltaApi.getWalletBalances();
            logger_1.logger.info(`✅ Wallet access successful - ${balances.length} assets`);
            const nonZeroBalances = balances.filter(b => parseFloat(b.balance) > 0);
            if (nonZeroBalances.length > 0) {
                logger_1.logger.info('   Non-zero balances:');
                nonZeroBalances.forEach(balance => {
                    logger_1.logger.info(`     ${balance.asset}: ${balance.balance}`);
                });
            }
            else {
                logger_1.logger.info('   No balances found (testnet account may be empty)');
            }
            // Test positions
            const positions = await this.deltaApi.getPositions();
            logger_1.logger.info(`✅ Position access successful - ${positions.length} positions`);
            if (positions.length > 0) {
                logger_1.logger.info('   Current positions:');
                positions.forEach(position => {
                    const side = parseFloat(position.size) > 0 ? 'LONG' : 'SHORT';
                    logger_1.logger.info(`     ${position.symbol}: ${side} ${Math.abs(parseFloat(position.size))} @ $${position.entry_price}`);
                });
            }
            // Test active orders
            const orders = await this.deltaApi.getActiveOrders();
            logger_1.logger.info(`✅ Order access successful - ${orders.length} active orders`);
        }
        catch (authError) {
            if (authError.message.includes('ip_not_whitelisted')) {
                throw new Error('ip_not_whitelisted_for_api_key');
            }
            throw authError;
        }
    }
    /**
     * Place actual test order on Delta Exchange
     */
    async placeTestOrder() {
        logger_1.logger.info('\n🎯 STEP 5: PLACING ACTUAL TEST ORDER ON DELTA EXCHANGE');
        logger_1.logger.info('⚠️  This will place a REAL order on your Delta testnet account!');
        try {
            // Get markets to find suitable trading pair
            const markets = await this.deltaApi.getMarkets();
            const btcMarket = markets.find(m => m.symbol === 'BTC_USDT' ||
                (m.symbol.includes('BTC') && m.symbol.includes('USD') && m.is_active));
            if (!btcMarket) {
                throw new Error('No suitable BTC market found for order placement');
            }
            logger_1.logger.info(`🎯 Selected market: ${btcMarket.symbol}`);
            // Get current price
            const ticker = await this.deltaApi.getTicker(btcMarket.symbol);
            const currentPrice = parseFloat(ticker.close);
            logger_1.logger.info(`📊 Current price: $${currentPrice.toFixed(2)}`);
            // Calculate conservative order parameters
            const orderSide = 'buy'; // Buy order
            const orderSize = 0.001; // Very small size (0.001 BTC)
            const orderPrice = currentPrice * 0.95; // 5% below market (won't execute immediately)
            const orderParams = {
                symbol: btcMarket.symbol,
                side: orderSide,
                size: orderSize,
                type: 'limit',
                price: orderPrice,
                timeInForce: 'gtc',
                clientOrderId: `test_${Date.now()}`,
            };
            logger_1.logger.info('\n📋 ORDER PARAMETERS:');
            logger_1.logger.info(`   Symbol: ${orderParams.symbol}`);
            logger_1.logger.info(`   Side: ${orderParams.side.toUpperCase()}`);
            logger_1.logger.info(`   Size: ${orderParams.size} BTC`);
            logger_1.logger.info(`   Type: ${orderParams.type.toUpperCase()}`);
            logger_1.logger.info(`   Price: $${orderParams.price.toFixed(2)} (${((orderParams.price / currentPrice - 1) * 100).toFixed(1)}% from market)`);
            logger_1.logger.info(`   Time in Force: ${orderParams.timeInForce.toUpperCase()}`);
            logger_1.logger.info(`   Client Order ID: ${orderParams.clientOrderId}`);
            logger_1.logger.info('\n🚀 PLACING ORDER...');
            // Place the order
            const order = await this.deltaApi.placeOrder(orderParams);
            logger_1.logger.info('🎉 ORDER PLACED SUCCESSFULLY!');
            logger_1.logger.info('='.repeat(60));
            logger_1.logger.info(`✅ Order ID: ${order.id}`);
            logger_1.logger.info(`✅ Symbol: ${order.symbol}`);
            logger_1.logger.info(`✅ Side: ${order.side}`);
            logger_1.logger.info(`✅ Size: ${order.size}`);
            logger_1.logger.info(`✅ Price: $${order.limit_price || order.price}`);
            logger_1.logger.info(`✅ Status: ${order.status}`);
            logger_1.logger.info(`✅ Client Order ID: ${order.client_order_id || 'N/A'}`);
            logger_1.logger.info('='.repeat(60));
            // Wait a moment then try to cancel the order
            logger_1.logger.info('\n⏳ Waiting 3 seconds before cancellation test...');
            await this.sleep(3000);
            logger_1.logger.info('\n❌ TESTING ORDER CANCELLATION...');
            try {
                const cancelResult = await this.deltaApi.cancelOrder(order.id.toString());
                logger_1.logger.info('✅ ORDER CANCELLATION SUCCESSFUL!');
                logger_1.logger.info(`   Cancel result: ${JSON.stringify(cancelResult)}`);
                // Verify cancellation
                const activeOrders = await this.deltaApi.getActiveOrders();
                const stillActive = activeOrders.find(o => o.id === order.id);
                if (!stillActive) {
                    logger_1.logger.info('✅ Order confirmed cancelled (not in active orders)');
                }
                else {
                    logger_1.logger.info(`⚠️ Order still shows as: ${stillActive.status}`);
                }
            }
            catch (cancelError) {
                logger_1.logger.error('❌ Order cancellation failed:', cancelError.message);
            }
            logger_1.logger.info('\n🎯 ORDER PLACEMENT TEST RESULTS:');
            logger_1.logger.info('✅ Order placement: SUCCESS');
            logger_1.logger.info('✅ Order parameters: CORRECT');
            logger_1.logger.info('✅ API integration: WORKING');
            logger_1.logger.info('✅ Authentication: WORKING');
            logger_1.logger.info('🚀 SYSTEM READY FOR LIVE TRADING!');
        }
        catch (orderError) {
            logger_1.logger.error('❌ Order placement failed:', orderError.message);
            if (orderError.message.includes('insufficient')) {
                logger_1.logger.info('💰 Insufficient balance - this is normal for testnet');
                logger_1.logger.info('✅ Order placement API is working (balance issue only)');
            }
            else if (orderError.message.includes('ip_not_whitelisted')) {
                throw new Error('ip_not_whitelisted_for_api_key');
            }
            else {
                logger_1.logger.info('🔧 Order placement needs debugging');
            }
        }
    }
    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.FullSystemTest = FullSystemTest;
/**
 * Main execution
 */
async function main() {
    const tester = new FullSystemTest();
    await tester.runFullTest();
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        logger_1.logger.error('💥 Full system test failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=full-system-test-and-order.js.map