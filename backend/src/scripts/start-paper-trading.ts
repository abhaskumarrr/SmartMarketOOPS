#!/usr/bin/env ts-node

/**
 * Enhanced Paper Trading Engine Startup Script
 * Starts paper trading with 75% balance allocation and frequency optimization
 */

import { PaperTradingEngine } from '../services/paperTradingEngine';
import { DeltaCredentials } from '../services/deltaExchangeService';
import { logger } from '../utils/logger';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function startEnhancedPaperTrading() {
  try {
    logger.info('\n🚀 STARTING ENHANCED PAPER TRADING ENGINE');
    logger.info('═'.repeat(80));
    logger.info('🎯 FREQUENCY OPTIMIZED TRADING WITH 75% BALANCE ALLOCATION');
    logger.info('⚡ TARGETING 3-5 PROFITABLE TRADES DAILY WITH 75%+ WIN RATE');
    logger.info('═'.repeat(80));

    // Delta Exchange credentials from environment
    const deltaCredentials: DeltaCredentials = {
      apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true' // Using testnet for paper trading
    };

    // Validate credentials
    if (!deltaCredentials.apiKey || !deltaCredentials.apiSecret) {
      throw new Error('❌ Delta Exchange API credentials not found in environment variables');
    }

    logger.info('✅ Delta Exchange credentials loaded');
    logger.info(`🔗 Using: ${deltaCredentials.testnet ? 'TESTNET' : 'PRODUCTION'} mode`);

    // Enhanced frequency-optimized configuration
    const config = {
      mlConfidenceThreshold: 80,        // 80%+ ML confidence required
      signalScoreThreshold: 72,         // 72+/100 signal score required
      qualityScoreThreshold: 78,        // 78+/100 quality score required
      targetTradesPerDay: 4,            // Target 3-5 trades daily
      targetWinRate: 75,                // Target 75% win rate
      mlAccuracy: 85,                   // 85% ML accuracy
      maxConcurrentTrades: 3,           // Max 3 concurrent trades
      balanceAllocationPercent: 75      // Use 75% of total balance
    };

    logger.info('\n⚡ FREQUENCY OPTIMIZATION CONFIGURATION:');
    logger.info(`   ML Confidence Threshold: ${config.mlConfidenceThreshold}%`);
    logger.info(`   Signal Score Threshold: ${config.signalScoreThreshold}/100`);
    logger.info(`   Quality Score Threshold: ${config.qualityScoreThreshold}/100`);
    logger.info(`   Target Trades Per Day: ${config.targetTradesPerDay}`);
    logger.info(`   Target Win Rate: ${config.targetWinRate}%`);
    logger.info(`   ML Accuracy: ${config.mlAccuracy}%`);
    logger.info(`   Max Concurrent Trades: ${config.maxConcurrentTrades}`);
    logger.info(`   Balance Allocation: ${config.balanceAllocationPercent}%`);

    // Initialize enhanced paper trading engine
    const paperTrader = new PaperTradingEngine(deltaCredentials, config);

    logger.info('\n🔄 Initializing enhanced paper trading engine...');

    // Start paper trading
    await paperTrader.startPaperTrading();

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n🛑 Received shutdown signal, stopping paper trading...');
      paperTrader.stopPaperTrading();
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      logger.info('\n🛑 Received termination signal, stopping paper trading...');
      paperTrader.stopPaperTrading();
      process.exit(0);
    });

  } catch (error) {
    logger.error('❌ Failed to start enhanced paper trading engine:', error);
    process.exit(1);
  }
}

// Start the enhanced paper trading engine
if (require.main === module) {
  startEnhancedPaperTrading().catch(error => {
    logger.error('❌ Unhandled error in paper trading startup:', error);
    process.exit(1);
  });
}

export { startEnhancedPaperTrading };
