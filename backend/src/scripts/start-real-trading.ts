#!/usr/bin/env node

/**
 * REAL TRADING ENGINE LAUNCHER
 * 
 * ⚠️  WARNING: This script places REAL ORDERS on Delta Exchange with REAL MONEY!
 * 
 * This is NOT a simulation - all trades will be executed live on your Delta Exchange account.
 * Make sure you understand the risks before running this script.
 */

import { RealTradingEngine } from '../services/realTradingEngine';
import { DeltaCredentials } from '../services/deltaExchangeService';
import { logger } from '../utils/logger';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

interface RealTradingConfig {
  balanceAllocationPercent: number;
  maxLeverage: number;
  riskPerTrade: number;
  targetTradesPerDay: number;
  targetWinRate: number;
  mlConfidenceThreshold: number;
  signalScoreThreshold: number;
  qualityScoreThreshold: number;
  maxDrawdownPercent: number;
  tradingAssets: string[];
  checkIntervalMs: number;
  progressReportIntervalMs: number;
}

async function startRealTrading() {
  try {
    logger.info('\n🚨 REAL TRADING ENGINE STARTUP');
    logger.info('=' .repeat(80));
    logger.info('⚠️  WARNING: This will place REAL ORDERS with REAL MONEY!');
    logger.info('💰 All trades will be executed live on Delta Exchange');
    logger.info('🚨 Make sure you understand the risks before proceeding');
    logger.info('=' .repeat(80));

    // Get Delta Exchange credentials
    const deltaCredentials: DeltaCredentials = {
      apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true' // Use testnet for safety
    };

    if (!deltaCredentials.apiKey || !deltaCredentials.apiSecret) {
      throw new Error('Delta Exchange API credentials not found in environment variables');
    }

    logger.info(`🔗 Delta Exchange Mode: ${deltaCredentials.testnet ? 'TESTNET' : 'PRODUCTION'}`);
    
    if (!deltaCredentials.testnet) {
      logger.warn('⚠️  PRODUCTION MODE: Using real money on live exchange!');
    } else {
      logger.info('✅ TESTNET MODE: Using test environment');
    }

    // Enhanced real trading configuration
    const config: RealTradingConfig = {
      balanceAllocationPercent: 75, // Use 75% of available balance
      maxLeverage: 100, // Delta Exchange testnet supports max 100x leverage
      riskPerTrade: 40, // High risk per trade (40% of balance)
      targetTradesPerDay: 4, // Target 4 trades per day
      targetWinRate: 75, // Target 75% win rate
      mlConfidenceThreshold: 80, // Require 80%+ ML confidence
      signalScoreThreshold: 72, // Require 72+ signal score
      qualityScoreThreshold: 78, // Require 78+ quality score
      maxDrawdownPercent: 20, // Stop if 20% drawdown
      tradingAssets: ['BTCUSD', 'ETHUSD'], // Trade BTC and ETH perpetuals
      checkIntervalMs: 30000, // Check every 30 seconds
      progressReportIntervalMs: 60000, // Report every 1 minute
    };

    logger.info('\n⚙️  REAL TRADING CONFIGURATION:');
    logger.info(`   Balance Allocation: ${config.balanceAllocationPercent}%`);
    logger.info(`   Max Leverage: ${config.maxLeverage}x (Delta Exchange testnet limit)`);
    logger.info(`   Risk per Trade: ${config.riskPerTrade}%`);
    logger.info(`   Target Trades/Day: ${config.targetTradesPerDay}`);
    logger.info(`   Target Win Rate: ${config.targetWinRate}%`);
    logger.info(`   ML Confidence Threshold: ${config.mlConfidenceThreshold}%`);
    logger.info(`   Signal Score Threshold: ${config.signalScoreThreshold}`);
    logger.info(`   Quality Score Threshold: ${config.qualityScoreThreshold}`);
    logger.info(`   Max Drawdown: ${config.maxDrawdownPercent}%`);
    logger.info(`   Trading Assets: ${config.tradingAssets.join(', ')}`);
    logger.info(`   💰 Will use ACTUAL testnet balance from your account`);

    // Initialize real trading engine
    const realTrader = new RealTradingEngine(deltaCredentials, config);

    logger.info('\n🔄 Initializing real trading engine...');
    logger.info('💰 This will fetch your REAL balance from Delta Exchange');
    logger.info('🚀 All subsequent trades will use REAL MONEY');

    // Start real trading
    await realTrader.startRealTrading();

  } catch (error) {
    logger.error('❌ Failed to start real trading:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  logger.info('\n🛑 Received SIGINT, shutting down real trading...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('\n🛑 Received SIGTERM, shutting down real trading...');
  process.exit(0);
});

// Start the real trading engine
if (require.main === module) {
  startRealTrading().catch(error => {
    logger.error('❌ Fatal error in real trading:', error);
    process.exit(1);
  });
}

export { startRealTrading };
