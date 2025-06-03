#!/usr/bin/env node
"use strict";
/**
 * Direct Trading Test - Bypass all complex logic and test Portfolio Manager directly
 */
Object.defineProperty(exports, "__esModule", { value: true });
const portfolioManager_1 = require("../services/portfolioManager");
const logger_1 = require("../utils/logger");
async function testDirectTrading() {
    logger_1.logger.info('🧪 DIRECT TRADING TEST - Portfolio Manager Isolation');
    // Create simple config
    const config = {
        symbol: 'BTCUSD',
        timeframe: '15m',
        startDate: new Date(),
        endDate: new Date(),
        initialCapital: 2000,
        leverage: 3,
        riskPerTrade: 2,
        commission: 0.1,
        slippage: 0.05,
        strategy: 'TEST',
        parameters: {},
    };
    // Create portfolio manager
    const portfolioManager = new portfolioManager_1.PortfolioManager(config);
    logger_1.logger.info('💼 Portfolio Manager initialized');
    logger_1.logger.info(`💰 Initial cash: $${portfolioManager.getCash()}`);
    logger_1.logger.info(`📊 Initial positions: ${portfolioManager.getPositions().length}`);
    // Create simple trading signal
    const signal = {
        id: 'test_signal_1',
        timestamp: Date.now(),
        symbol: 'BTCUSD',
        type: 'BUY',
        price: 100000,
        quantity: 0.01, // Small quantity
        confidence: 80,
        strategy: 'TEST',
        reason: 'Direct test signal',
        stopLoss: 98000,
        takeProfit: 102000,
        riskReward: 2.0,
    };
    logger_1.logger.info('🔥 Testing signal:', {
        type: signal.type,
        symbol: signal.symbol,
        price: signal.price,
        quantity: signal.quantity,
        confidence: signal.confidence,
    });
    // Test 1: Execute trade
    logger_1.logger.info('\n🧪 TEST 1: Execute BUY trade');
    const trade1 = portfolioManager.executeTrade(signal, signal.price, signal.timestamp);
    if (trade1) {
        logger_1.logger.info('✅ Trade 1 SUCCESS:', {
            id: trade1.id,
            side: trade1.side,
            quantity: trade1.quantity,
            entryPrice: trade1.entryPrice,
            commission: trade1.commission,
        });
    }
    else {
        logger_1.logger.error('❌ Trade 1 FAILED: executeTrade returned null');
    }
    logger_1.logger.info(`💰 Cash after trade 1: $${portfolioManager.getCash()}`);
    logger_1.logger.info(`📊 Positions after trade 1: ${portfolioManager.getPositions().length}`);
    logger_1.logger.info(`📈 Trades recorded: ${portfolioManager.getTrades().length}`);
    // Test 2: Try another trade
    const signal2 = {
        ...signal,
        id: 'test_signal_2',
        type: 'SELL',
        price: 101000,
        quantity: 0.005,
    };
    logger_1.logger.info('\n🧪 TEST 2: Execute SELL trade');
    const trade2 = portfolioManager.executeTrade(signal2, signal2.price, signal2.timestamp + 1000);
    if (trade2) {
        logger_1.logger.info('✅ Trade 2 SUCCESS:', {
            id: trade2.id,
            side: trade2.side,
            quantity: trade2.quantity,
            entryPrice: trade2.entryPrice,
            commission: trade2.commission,
        });
    }
    else {
        logger_1.logger.error('❌ Trade 2 FAILED: executeTrade returned null');
    }
    logger_1.logger.info(`💰 Cash after trade 2: $${portfolioManager.getCash()}`);
    logger_1.logger.info(`📊 Positions after trade 2: ${portfolioManager.getPositions().length}`);
    logger_1.logger.info(`📈 Trades recorded: ${portfolioManager.getTrades().length}`);
    // Test 3: Very small trade
    const signal3 = {
        ...signal,
        id: 'test_signal_3',
        type: 'BUY',
        price: 100000,
        quantity: 0.001, // Very small
    };
    logger_1.logger.info('\n🧪 TEST 3: Execute very small BUY trade');
    const trade3 = portfolioManager.executeTrade(signal3, signal3.price, signal3.timestamp + 2000);
    if (trade3) {
        logger_1.logger.info('✅ Trade 3 SUCCESS:', {
            id: trade3.id,
            side: trade3.side,
            quantity: trade3.quantity,
            entryPrice: trade3.entryPrice,
            commission: trade3.commission,
        });
    }
    else {
        logger_1.logger.error('❌ Trade 3 FAILED: executeTrade returned null');
    }
    // Final status
    logger_1.logger.info('\n📊 FINAL STATUS:');
    logger_1.logger.info(`💰 Final cash: $${portfolioManager.getCash()}`);
    logger_1.logger.info(`📊 Final positions: ${portfolioManager.getPositions().length}`);
    logger_1.logger.info(`📈 Total trades: ${portfolioManager.getTrades().length}`);
    const positions = portfolioManager.getPositions();
    if (positions.length > 0) {
        logger_1.logger.info('📍 Active positions:');
        positions.forEach(pos => {
            logger_1.logger.info(`   ${pos.symbol} ${pos.side}: ${pos.size} @ $${pos.entryPrice}`);
        });
    }
    const trades = portfolioManager.getTrades();
    if (trades.length > 0) {
        logger_1.logger.info('📋 Completed trades:');
        trades.forEach(trade => {
            logger_1.logger.info(`   ${trade.symbol} ${trade.side}: ${trade.quantity} @ $${trade.entryPrice} -> $${trade.exitPrice} (P&L: $${trade.pnl})`);
        });
    }
    // Test calculations
    logger_1.logger.info('\n🧮 CALCULATION TEST:');
    const testPrice = 100000;
    const testQuantity = 0.01;
    const cost = testQuantity * testPrice; // $1000
    const commission = cost * (config.commission / 100); // $1
    const totalCost = cost + commission; // $1001
    const requiredCash = totalCost / config.leverage; // $333.67
    logger_1.logger.info(`Test calculation for 0.01 BTC at $100,000:`);
    logger_1.logger.info(`  Cost: $${cost}`);
    logger_1.logger.info(`  Commission: $${commission}`);
    logger_1.logger.info(`  Total cost: $${totalCost}`);
    logger_1.logger.info(`  Required cash (3x leverage): $${requiredCash.toFixed(2)}`);
    logger_1.logger.info(`  Available cash: $${portfolioManager.getCash()}`);
    logger_1.logger.info(`  Can afford: ${requiredCash <= portfolioManager.getCash() ? 'YES' : 'NO'}`);
    logger_1.logger.info('\n🎉 Direct trading test completed');
}
// Run the test
testDirectTrading().catch(console.error);
//# sourceMappingURL=test-direct-trading.js.map