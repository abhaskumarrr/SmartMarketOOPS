#!/usr/bin/env node
"use strict";
/**
 * Simple Backtest Test
 */
Object.defineProperty(exports, "__esModule", { value: true });
console.log('🚀 Starting Simple Backtest Test');
// Test the IntelligentTradingBotBacktester
const IntelligentTradingBotBacktester_1 = require("../backtesting/IntelligentTradingBotBacktester");
async function testBacktest() {
    console.log('📋 Initializing backtest configuration...');
    const config = {
        symbol: 'ETHUSD',
        startDate: '2023-01-01',
        endDate: '2023-01-31', // Just test 1 month
        initialCapital: 10,
        leverage: 200,
        riskPerTrade: 40,
        maxPositions: 1,
        timeframe: '1h'
    };
    console.log('💰 Initial Capital:', config.initialCapital);
    console.log('⚡ Leverage:', config.leverage + 'x');
    console.log('🎯 Risk Per Trade:', config.riskPerTrade + '%');
    try {
        console.log('🔧 Creating backtester instance...');
        const backtester = new IntelligentTradingBotBacktester_1.IntelligentTradingBotBacktester(config);
        console.log('▶️ Running backtest...');
        const results = await backtester.runBacktest();
        console.log('✅ Backtest completed!');
        console.log('📊 Results Summary:');
        console.log('- Total Trades:', results.summary.totalTrades);
        console.log('- Win Rate:', results.summary.winRate.toFixed(1) + '%');
        console.log('- Final Balance: $' + results.summary.finalBalance.toFixed(2));
        console.log('- Total Return:', results.summary.totalReturnPercent.toFixed(1) + '%');
    }
    catch (error) {
        console.error('❌ Backtest failed:', error);
    }
}
// Execute test
testBacktest().then(() => {
    console.log('🏁 Test completed');
}).catch(error => {
    console.error('💥 Test failed:', error);
});
//# sourceMappingURL=test-backtest.js.map