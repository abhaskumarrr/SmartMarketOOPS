#!/usr/bin/env node
/**
 * Delta Exchange Paper Trading System
 * Safe paper trading with real market data from Delta Exchange India testnet
 */

import { DeltaExchangeUnified } from '../src/services/DeltaExchangeUnified';
import { logger } from '../src/utils/logger';

interface PaperPosition {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  currentPrice: number;
  leverage: number;
  pnl: number;
  pnlPercentage: number;
  entryTime: number;
  stopLoss?: number;
  takeProfit?: number;
}

interface PaperPortfolio {
  initialBalance: number;
  currentBalance: number;
  availableBalance: number;
  totalPnl: number;
  totalReturn: number;
  positions: Map<string, PaperPosition>;
  tradeHistory: any[];
}

class DeltaPaperTradingBot {
  private deltaService: DeltaExchangeUnified;
  private portfolio: PaperPortfolio;
  private config = {
    symbols: ['BTCUSD', 'ETHUSD'],
    initialBalance: 1000, // $1000 paper money
    maxLeverage: { 'BTCUSD': 50, 'ETHUSD': 50 },
    riskPerTrade: 10, // 10% risk per trade
    maxConcurrentTrades: 2,
    tradingInterval: 15000, // 15 seconds
    stopLossPercentage: 3, // 3% stop loss
    takeProfitRatio: 2 // 2:1 risk/reward
  };
  private isRunning = false;
  private tradeCounter = 0;

  constructor() {
    // Initialize Delta Exchange service for market data
    this.deltaService = new DeltaExchangeUnified({
      apiKey: process.env.DELTA_EXCHANGE_API_KEY!,
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET!,
      testnet: true
    });

    // Initialize paper portfolio
    this.portfolio = {
      initialBalance: this.config.initialBalance,
      currentBalance: this.config.initialBalance,
      availableBalance: this.config.initialBalance,
      totalPnl: 0,
      totalReturn: 0,
      positions: new Map(),
      tradeHistory: []
    };
  }

  /**
   * Initialize the paper trading system
   */
  async initialize(): Promise<void> {
    try {
      logger.info('🚀 Initializing Delta Exchange Paper Trading System...');

      // Wait for Delta service to be ready
      let retries = 0;
      while (!this.deltaService.isReady() && retries < 10) {
        logger.info(`⏳ Waiting for Delta Exchange service... (${retries + 1}/10)`);
        await this.sleep(2000);
        retries++;
      }

      if (!this.deltaService.isReady()) {
        throw new Error('Delta Exchange service failed to initialize');
      }

      logger.info('✅ Delta Exchange service connected successfully');
      this.displayConfiguration();

    } catch (error) {
      logger.error('❌ Failed to initialize paper trading system:', error);
      throw error;
    }
  }

  /**
   * Display trading configuration
   */
  displayConfiguration(): void {
    logger.info('\n📊 DELTA EXCHANGE PAPER TRADING SYSTEM');
    logger.info('═'.repeat(60));
    logger.info(`💰 Paper Balance: $${this.config.initialBalance}`);
    logger.info(`📊 Trading Symbols: ${this.config.symbols.join(', ')}`);
    logger.info(`⚡ Max Leverage: BTC=${this.config.maxLeverage.BTCUSD}x, ETH=${this.config.maxLeverage.ETHUSD}x`);
    logger.info(`🎯 Risk Per Trade: ${this.config.riskPerTrade}%`);
    logger.info(`🛡️ Stop Loss: ${this.config.stopLossPercentage}%`);
    logger.info(`🎯 Take Profit Ratio: ${this.config.takeProfitRatio}:1`);
    logger.info(`🔄 Trading Interval: ${this.config.tradingInterval / 1000}s`);
    logger.info(`🏢 Exchange: Delta Exchange India (Real Market Data)`);
    logger.info(`📝 Mode: PAPER TRADING (No Real Money)`);
    logger.info('═'.repeat(60));
  }

  /**
   * Generate simple trading signal
   */
  async generateTradingSignal(symbol: string): Promise<{ side: 'buy' | 'sell'; confidence: number } | null> {
    try {
      const marketData = await this.deltaService.getMarketData(symbol);
      const currentPrice = marketData.last_price;

      // Simple momentum strategy for paper trading
      const random = Math.random();
      
      // Generate signals based on price levels and some randomness
      if (symbol === 'BTCUSD') {
        if (currentPrice < 105000 && random > 0.6) {
          return { side: 'buy', confidence: 0.8 };
        } else if (currentPrice > 105500 && random > 0.6) {
          return { side: 'sell', confidence: 0.8 };
        }
      }

      if (symbol === 'ETHUSD') {
        if (currentPrice < 2600 && random > 0.6) {
          return { side: 'buy', confidence: 0.8 };
        } else if (currentPrice > 2650 && random > 0.6) {
          return { side: 'sell', confidence: 0.8 };
        }
      }

      return null;
    } catch (error) {
      logger.error(`❌ Error generating signal for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Calculate position size for paper trading
   */
  calculatePositionSize(symbol: string, entryPrice: number): number {
    const riskAmount = this.portfolio.availableBalance * (this.config.riskPerTrade / 100);
    const stopLossDistance = entryPrice * (this.config.stopLossPercentage / 100);
    const leverage = this.config.maxLeverage[symbol] || 50;
    
    // Calculate position size in USD value
    const positionValue = (riskAmount / stopLossDistance) * entryPrice * leverage;
    
    // Convert to contract units
    return Math.floor(positionValue);
  }

  /**
   * Open a paper position
   */
  async openPaperPosition(symbol: string, side: 'buy' | 'sell', size: number, entryPrice: number): Promise<void> {
    this.tradeCounter++;
    const positionId = `${symbol}_${this.tradeCounter}`;
    const leverage = this.config.maxLeverage[symbol] || 50;

    const position: PaperPosition = {
      id: positionId,
      symbol,
      side,
      size,
      entryPrice,
      currentPrice: entryPrice,
      leverage,
      pnl: 0,
      pnlPercentage: 0,
      entryTime: Date.now(),
      stopLoss: side === 'buy' 
        ? entryPrice * (1 - this.config.stopLossPercentage / 100)
        : entryPrice * (1 + this.config.stopLossPercentage / 100),
      takeProfit: side === 'buy'
        ? entryPrice * (1 + (this.config.stopLossPercentage * this.config.takeProfitRatio) / 100)
        : entryPrice * (1 - (this.config.stopLossPercentage * this.config.takeProfitRatio) / 100)
    };

    this.portfolio.positions.set(positionId, position);
    
    // Update available balance (margin requirement)
    const marginRequired = (size * entryPrice) / leverage;
    this.portfolio.availableBalance -= marginRequired;

    logger.info(`📝 Paper Position Opened:`);
    logger.info(`   ${side.toUpperCase()} ${size} ${symbol} @ $${entryPrice.toFixed(2)}`);
    logger.info(`   Stop Loss: $${position.stopLoss?.toFixed(2)}`);
    logger.info(`   Take Profit: $${position.takeProfit?.toFixed(2)}`);
    logger.info(`   Leverage: ${leverage}x`);
  }

  /**
   * Update all positions with current market prices
   */
  async updatePositions(): Promise<void> {
    for (const [positionId, position] of this.portfolio.positions) {
      try {
        const marketData = await this.deltaService.getMarketData(position.symbol);
        const currentPrice = marketData.last_price;
        
        position.currentPrice = currentPrice;
        
        // Calculate PnL
        const priceDiff = position.side === 'buy' 
          ? currentPrice - position.entryPrice
          : position.entryPrice - currentPrice;
        
        position.pnl = (priceDiff / position.entryPrice) * position.size * position.leverage;
        position.pnlPercentage = (priceDiff / position.entryPrice) * 100 * position.leverage;

        // Check stop loss and take profit
        if (this.shouldClosePosition(position)) {
          await this.closePaperPosition(positionId, 'auto');
        }

      } catch (error) {
        logger.error(`❌ Error updating position ${positionId}:`, error);
      }
    }
  }

  /**
   * Check if position should be closed
   */
  shouldClosePosition(position: PaperPosition): boolean {
    if (position.side === 'buy') {
      return position.currentPrice <= (position.stopLoss || 0) || 
             position.currentPrice >= (position.takeProfit || Infinity);
    } else {
      return position.currentPrice >= (position.stopLoss || Infinity) || 
             position.currentPrice <= (position.takeProfit || 0);
    }
  }

  /**
   * Close a paper position
   */
  async closePaperPosition(positionId: string, reason: string): Promise<void> {
    const position = this.portfolio.positions.get(positionId);
    if (!position) return;

    // Update portfolio
    this.portfolio.totalPnl += position.pnl;
    this.portfolio.currentBalance += position.pnl;
    
    // Free up margin
    const marginRequired = (position.size * position.entryPrice) / position.leverage;
    this.portfolio.availableBalance += marginRequired;

    // Add to trade history
    this.portfolio.tradeHistory.push({
      ...position,
      closeTime: Date.now(),
      closeReason: reason,
      duration: Date.now() - position.entryTime
    });

    this.portfolio.positions.delete(positionId);

    logger.info(`📝 Paper Position Closed (${reason}):`);
    logger.info(`   ${position.side.toUpperCase()} ${position.size} ${position.symbol}`);
    logger.info(`   Entry: $${position.entryPrice.toFixed(2)} → Exit: $${position.currentPrice.toFixed(2)}`);
    logger.info(`   PnL: $${position.pnl.toFixed(2)} (${position.pnlPercentage.toFixed(2)}%)`);
  }

  /**
   * Main paper trading loop
   */
  async startPaperTrading(): Promise<void> {
    this.isRunning = true;
    logger.info('🚀 Starting paper trading loop...');

    let iteration = 0;
    const maxIterations = 100; // Limit for demo

    while (this.isRunning && iteration < maxIterations) {
      try {
        iteration++;
        logger.info(`\n🔄 Paper Trading Iteration ${iteration}`);
        
        // Update existing positions
        await this.updatePositions();
        
        // Look for new trading opportunities
        if (this.portfolio.positions.size < this.config.maxConcurrentTrades) {
          for (const symbol of this.config.symbols) {
            // Skip if we already have a position for this symbol
            const hasPosition = Array.from(this.portfolio.positions.values())
              .some(pos => pos.symbol === symbol);
            
            if (hasPosition) continue;

            // Generate trading signal
            const signal = await this.generateTradingSignal(symbol);
            
            if (signal && signal.confidence > 0.7) {
              const marketData = await this.deltaService.getMarketData(symbol);
              const currentPrice = marketData.last_price;
              const positionSize = this.calculatePositionSize(symbol, currentPrice);
              
              if (positionSize > 0 && this.portfolio.availableBalance > 100) {
                await this.openPaperPosition(symbol, signal.side, positionSize, currentPrice);
              }
            }
          }
        }

        // Display current status
        this.displayPaperStatus();
        
        // Wait for next iteration
        logger.info(`⏳ Waiting ${this.config.tradingInterval / 1000}s for next iteration...`);
        await this.sleep(this.config.tradingInterval);
        
      } catch (error) {
        logger.error('❌ Error in paper trading loop:', error);
        await this.sleep(5000);
      }
    }

    logger.info('🏁 Paper trading completed');
    this.generatePaperReport();
  }

  /**
   * Display current paper trading status
   */
  displayPaperStatus(): void {
    this.portfolio.totalReturn = ((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance) * 100;
    
    logger.info('\n📊 PAPER TRADING STATUS:');
    logger.info(`💰 Balance: $${this.portfolio.currentBalance.toFixed(2)} (${this.portfolio.totalReturn.toFixed(2)}%)`);
    logger.info(`📈 Total PnL: $${this.portfolio.totalPnl.toFixed(2)}`);
    logger.info(`🎯 Active Positions: ${this.portfolio.positions.size}`);
    logger.info(`📊 Total Trades: ${this.portfolio.tradeHistory.length}`);
    
    // Show active positions
    if (this.portfolio.positions.size > 0) {
      logger.info('📋 Active Positions:');
      for (const position of this.portfolio.positions.values()) {
        logger.info(`   ${position.symbol}: ${position.side} ${position.size} @ $${position.entryPrice.toFixed(2)} | PnL: $${position.pnl.toFixed(2)}`);
      }
    }
  }

  /**
   * Generate final paper trading report
   */
  generatePaperReport(): void {
    const winningTrades = this.portfolio.tradeHistory.filter(t => t.pnl > 0).length;
    const losingTrades = this.portfolio.tradeHistory.filter(t => t.pnl < 0).length;
    const winRate = this.portfolio.tradeHistory.length > 0 ? (winningTrades / this.portfolio.tradeHistory.length) * 100 : 0;
    
    logger.info('\n📊 PAPER TRADING FINAL REPORT');
    logger.info('═'.repeat(60));
    logger.info(`💰 Starting Balance: $${this.portfolio.initialBalance}`);
    logger.info(`💰 Final Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
    logger.info(`📈 Total Return: ${this.portfolio.totalReturn.toFixed(2)}%`);
    logger.info(`📊 Total Trades: ${this.portfolio.tradeHistory.length}`);
    logger.info(`✅ Winning Trades: ${winningTrades}`);
    logger.info(`❌ Losing Trades: ${losingTrades}`);
    logger.info(`🎯 Win Rate: ${winRate.toFixed(1)}%`);
    logger.info(`🏢 Exchange: Delta Exchange India (Paper Trading)`);
    logger.info('═'.repeat(60));
  }

  /**
   * Stop paper trading
   */
  stop(): void {
    this.isRunning = false;
    logger.info('🛑 Paper trading stopped');
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Main execution
async function main() {
  const bot = new DeltaPaperTradingBot();
  
  try {
    await bot.initialize();
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n🛑 Received SIGINT, shutting down gracefully...');
      bot.stop();
      process.exit(0);
    });
    
    await bot.startPaperTrading();
    
  } catch (error) {
    logger.error('❌ Paper trading failed:', error);
    process.exit(1);
  }
}

// Run the bot
if (require.main === module) {
  main().catch(console.error);
}

export { DeltaPaperTradingBot };
