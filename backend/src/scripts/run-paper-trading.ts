#!/usr/bin/env node

/**
 * Paper Trading System Runner
 * Runs live paper trading with dynamic take profit system
 */

import { PaperTradingEngine } from '../services/paperTradingEngine';
import { logger } from '../utils/logger';

class PaperTradingRunner {
  private engine: PaperTradingEngine;
  private statusInterval: NodeJS.Timeout | null = null;

  constructor() {
    // Initialize with Delta Exchange credentials and configuration
    const deltaCredentials = {
      apiKey: process.env.DELTA_API_KEY || 'test-key',
      apiSecret: process.env.DELTA_SECRET_KEY || 'test-secret',
      testnet: true
    };

    const config = {
      mlConfidenceThreshold: 80, // 80%+ ML confidence
      signalScoreThreshold: 72, // 72+/100 signal score
      qualityScoreThreshold: 78, // 78+/100 quality score
      targetTradesPerDay: 4, // Target 3-5 trades daily
      targetWinRate: 75, // Target 75% win rate
      mlAccuracy: 85, // 85% ML accuracy
      maxConcurrentTrades: 3, // Max 3 concurrent trades
      balanceAllocationPercent: 75, // Use 75% of total balance
    };

    this.engine = new PaperTradingEngine(deltaCredentials, config);
  }

  /**
   * Start paper trading system
   */
  public async startPaperTrading(): Promise<void> {
    try {
      logger.info('🚀 STARTING PAPER TRADING SYSTEM');
      logger.info('=' .repeat(80));
      
      logger.info('📊 PAPER TRADING CONFIGURATION:');
      logger.info('   💰 Initial Balance: $2,000 (Delta Exchange spot trading)');
      logger.info('   ⚡ Leverage: 3x ($6,000 max buying power)');
      logger.info('   🎯 Risk Per Trade: 2% ($40 max risk per trade)');
      logger.info('   📈 Assets: BTC/USDT, ETH/USDT (Delta Exchange spot pairs)');
      logger.info('   🔄 Strategy: Enhanced Dynamic Take Profit System');
      logger.info('   📊 Features: Real market data, market regime detection, partial exits, trailing stops');
      logger.info('   🏢 Exchange: Delta Exchange (Testnet)');

      // Start status monitoring
      this.startStatusMonitoring();

      // Start the paper trading engine
      await this.engine.startPaperTrading();

    } catch (error) {
      logger.error('❌ Paper trading failed:', error);
      this.stopStatusMonitoring();
      throw error;
    }
  }

  /**
   * Stop paper trading system
   */
  public stopPaperTrading(): void {
    logger.info('🛑 Stopping paper trading system...');
    this.engine.stopPaperTrading();
    this.stopStatusMonitoring();
  }

  /**
   * Start status monitoring
   */
  private startStatusMonitoring(): void {
    this.statusInterval = setInterval(() => {
      this.displayStatus();
    }, 30000); // Every 30 seconds
  }

  /**
   * Stop status monitoring
   */
  private stopStatusMonitoring(): void {
    if (this.statusInterval) {
      clearInterval(this.statusInterval);
      this.statusInterval = null;
    }
  }

  /**
   * Display current status
   */
  private displayStatus(): void {
    const portfolio = this.engine.getPortfolioStatus();
    const activeTrades = this.engine.getActiveTrades();
    const closedTrades = this.engine.getClosedTrades();

    logger.info('\n📊 PAPER TRADING STATUS UPDATE:');
    logger.info(`   💰 Current Balance: $${portfolio.currentBalance.toFixed(2)}`);
    logger.info(`   📈 Total P&L: $${portfolio.totalPnl.toFixed(2)} (${((portfolio.currentBalance - portfolio.initialBalance) / portfolio.initialBalance * 100).toFixed(2)}%)`);
    logger.info(`   📊 Active Trades: ${activeTrades.length}`);
    logger.info(`   ✅ Closed Trades: ${closedTrades.length}`);
    logger.info(`   🎯 Win Rate: ${portfolio.winRate.toFixed(1)}%`);
    logger.info(`   ⚠️ Current Drawdown: ${portfolio.currentDrawdown.toFixed(2)}%`);

    // Show active trades
    if (activeTrades.length > 0) {
      logger.info('   🔥 Active Trades:');
      activeTrades.forEach(trade => {
        const unrealizedPercent = ((trade.unrealizedPnl / portfolio.initialBalance) * 100);
        logger.info(`     ${trade.symbol}: ${trade.side} $${trade.unrealizedPnl.toFixed(2)} (${unrealizedPercent.toFixed(2)}%) - ${trade.partialExits.length} exits`);
      });
    }

    // Show recent closed trades
    if (closedTrades.length > 0) {
      const recentTrades = closedTrades.slice(-3); // Last 3 trades
      logger.info('   📋 Recent Closed Trades:');
      recentTrades.forEach(trade => {
        const pnlPercent = ((trade.pnl! / portfolio.initialBalance) * 100);
        const status = trade.pnl! > 0 ? '✅' : '❌';
        logger.info(`     ${status} ${trade.symbol}: $${trade.pnl!.toFixed(2)} (${pnlPercent.toFixed(2)}%) - ${trade.reason}`);
      });
    }
  }

  /**
   * Handle graceful shutdown
   */
  public setupGracefulShutdown(): void {
    const shutdown = () => {
      logger.info('\n🛑 Received shutdown signal, stopping paper trading...');
      this.stopPaperTrading();
      process.exit(0);
    };

    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    process.on('SIGUSR2', shutdown); // For nodemon
  }
}

/**
 * Main execution function
 */
async function main() {
  const runner = new PaperTradingRunner();
  
  try {
    // Setup graceful shutdown
    runner.setupGracefulShutdown();
    
    // Start paper trading
    await runner.startPaperTrading();
    
  } catch (error) {
    logger.error('💥 Paper trading runner failed:', error);
    process.exit(1);
  }
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { PaperTradingRunner };
