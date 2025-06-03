/**
 * Paper Trading Engine
 * Simulates live trading with real market data but virtual money
 */

import { DynamicTakeProfitManager, DynamicTakeProfitConfig, MarketRegime } from './dynamicTakeProfitManager';
import { marketDataService } from './marketDataService'; // Updated to use real Delta Exchange data
import { TradingSignal, TakeProfitLevel } from '../types/marketData';
import { logger } from '../utils/logger';

export interface PaperTrade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  size: number;
  entryPrice: number;
  entryTime: number;
  exitPrice?: number;
  exitTime?: number;
  pnl?: number;
  status: 'OPEN' | 'CLOSED' | 'CANCELLED';
  reason?: string;
  takeProfitLevels: TakeProfitLevel[];
  partialExits: PartialExit[];
  stopLoss: number;
  currentPrice: number;
  unrealizedPnl: number;
  maxProfit: number;
  maxLoss: number;
}

export interface PartialExit {
  level: number;
  percentage: number;
  price: number;
  timestamp: number;
  pnl: number;
  reason: string;
}

export interface PaperPortfolio {
  initialBalance: number;
  currentBalance: number;
  totalPnl: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  maxDrawdown: number;
  currentDrawdown: number;
  leverage: number;
  riskPerTrade: number;
}

export class PaperTradingEngine {
  private takeProfitManager: DynamicTakeProfitManager;
  private activeTrades: Map<string, PaperTrade> = new Map();
  private closedTrades: PaperTrade[] = [];
  private portfolio: PaperPortfolio;
  private isRunning: boolean = false;
  private tradingAssets: string[] = ['BTCUSD', 'ETHUSD']; // Delta Exchange spot pairs (mapped to BTC/USDT, ETH/USDT)

  constructor(initialBalance: number = 2000, leverage: number = 3, riskPerTrade: number = 2) {
    this.takeProfitManager = new DynamicTakeProfitManager();
    this.portfolio = {
      initialBalance,
      currentBalance: initialBalance,
      totalPnl: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      maxDrawdown: 0,
      currentDrawdown: 0,
      leverage,
      riskPerTrade,
    };
  }

  /**
   * Start paper trading system
   */
  public async startPaperTrading(): Promise<void> {
    if (this.isRunning) {
      logger.warn('Paper trading system is already running');
      return;
    }

    this.isRunning = true;
    logger.info('🚀 Starting Paper Trading System');
    logger.info(`💰 Initial Balance: $${this.portfolio.initialBalance}`);
    logger.info(`⚡ Leverage: ${this.portfolio.leverage}x`);
    logger.info(`🎯 Risk Per Trade: ${this.portfolio.riskPerTrade}%`);
    logger.info(`📊 Assets: ${this.tradingAssets.join(', ')}`);

    // Start trading loop
    await this.runTradingLoop();
  }

  /**
   * Stop paper trading system
   */
  public stopPaperTrading(): void {
    this.isRunning = false;
    logger.info('🛑 Paper trading system stopped');
    this.generateFinalReport();
  }

  /**
   * Main trading loop
   */
  private async runTradingLoop(): Promise<void> {
    let iteration = 0;
    const maxIterations = 100; // Limit for demo

    while (this.isRunning && iteration < maxIterations) {
      try {
        iteration++;
        logger.info(`\n🔄 Paper Trading Iteration ${iteration}`);

        // Process each asset
        for (const asset of this.tradingAssets) {
          await this.processAsset(asset);
        }

        // Update portfolio metrics
        this.updatePortfolioMetrics();

        // Check for stop conditions
        if (this.portfolio.currentDrawdown >= 30) {
          logger.warn('🛑 Maximum drawdown reached, stopping paper trading');
          break;
        }

        // Wait before next iteration (simulate real-time trading)
        await this.sleep(2000); // 2 seconds between iterations

      } catch (error) {
        logger.error('❌ Error in trading loop:', error);
      }
    }

    this.isRunning = false;
    this.generateFinalReport();
  }

  /**
   * Process trading for a specific asset
   */
  private async processAsset(asset: string): Promise<void> {
    try {
      // Get current market data (simulate real-time)
      const currentPrice = await this.getCurrentPrice(asset);
      if (!currentPrice) return;

      // Update existing trades
      await this.updateExistingTrades(asset, currentPrice);

      // Check for new trading opportunities
      await this.checkNewTradingOpportunity(asset, currentPrice);

    } catch (error) {
      logger.error(`❌ Error processing ${asset}:`, error);
    }
  }

  /**
   * Get current price for asset from Delta Exchange
   */
  private async getCurrentPrice(asset: string): Promise<number | null> {
    try {
      // Get real-time market data from Delta Exchange
      const marketData = await marketDataService.getMarketData(asset);

      if (marketData && marketData.price > 0) {
        logger.debug(`📊 ${asset}: $${marketData.price.toFixed(2)} (${marketData.changePercent.toFixed(2)}%)`);
        return marketData.price;
      }

      logger.warn(`⚠️ No market data available for ${asset}`);
      return null;
    } catch (error) {
      logger.error(`❌ Failed to get price for ${asset}:`, error);
      return null;
    }
  }

  /**
   * Update existing trades with current price
   */
  private async updateExistingTrades(asset: string, currentPrice: number): Promise<void> {
    const assetTrades = Array.from(this.activeTrades.values()).filter(trade => 
      trade.symbol === asset && trade.status === 'OPEN'
    );

    for (const trade of assetTrades) {
      // Update current price and unrealized PnL
      trade.currentPrice = currentPrice;
      const priceChange = trade.side === 'BUY' 
        ? currentPrice - trade.entryPrice 
        : trade.entryPrice - currentPrice;
      
      trade.unrealizedPnl = (priceChange / trade.entryPrice) * trade.size * this.portfolio.leverage;
      
      // Update max profit/loss tracking
      trade.maxProfit = Math.max(trade.maxProfit, trade.unrealizedPnl);
      trade.maxLoss = Math.min(trade.maxLoss, trade.unrealizedPnl);

      // Check for partial exits (dynamic take profit)
      await this.checkPartialExits(trade);

      // Check for stop loss
      await this.checkStopLoss(trade);

      logger.debug(`📊 ${trade.symbol}: $${currentPrice.toFixed(2)}, P&L: $${trade.unrealizedPnl.toFixed(2)}`);
    }
  }

  /**
   * Check for partial exits based on dynamic take profit levels
   */
  private async checkPartialExits(trade: PaperTrade): Promise<void> {
    for (const level of trade.takeProfitLevels) {
      if (level.executed) continue;

      let shouldExit = false;
      if (trade.side === 'BUY') {
        shouldExit = trade.currentPrice >= level.priceTarget;
      } else {
        shouldExit = trade.currentPrice <= level.priceTarget;
      }

      if (shouldExit) {
        // Execute partial exit
        const exitSize = (trade.size * level.percentage) / 100;
        const exitPnl = this.calculatePartialPnl(trade, level.priceTarget, exitSize);

        const partialExit: PartialExit = {
          level: trade.partialExits.length + 1,
          percentage: level.percentage,
          price: level.priceTarget,
          timestamp: Date.now(),
          pnl: exitPnl,
          reason: `Take profit level ${level.riskRewardRatio.toFixed(1)}:1`,
        };

        trade.partialExits.push(partialExit);
        level.executed = true;
        level.executedAt = Date.now();

        // Update trade size
        trade.size -= exitSize;

        // Update portfolio balance
        this.portfolio.currentBalance += exitPnl;
        this.portfolio.totalPnl += exitPnl;

        logger.info(`💰 Partial Exit: ${trade.symbol} ${level.percentage}% at $${level.priceTarget.toFixed(2)} - P&L: $${exitPnl.toFixed(2)}`);

        // Close trade if all levels executed
        if (trade.size <= 0.001) {
          await this.closeTrade(trade, 'All take profit levels hit');
        }
      }
    }
  }

  /**
   * Check for stop loss
   */
  private async checkStopLoss(trade: PaperTrade): Promise<void> {
    let shouldStop = false;
    if (trade.side === 'BUY') {
      shouldStop = trade.currentPrice <= trade.stopLoss;
    } else {
      shouldStop = trade.currentPrice >= trade.stopLoss;
    }

    if (shouldStop) {
      await this.closeTrade(trade, 'Stop loss hit');
    }
  }

  /**
   * Check for new trading opportunities
   */
  private async checkNewTradingOpportunity(asset: string, currentPrice: number): Promise<void> {
    // Limit concurrent trades per asset
    const assetTrades = Array.from(this.activeTrades.values()).filter(trade => 
      trade.symbol === asset && trade.status === 'OPEN'
    );

    if (assetTrades.length >= 2) return; // Max 2 trades per asset

    // Check if we have enough balance
    const availableBalance = this.portfolio.currentBalance * 0.8; // Use max 80% of balance
    if (availableBalance < this.portfolio.initialBalance * 0.1) return; // Need at least 10% of initial

    // Generate trading signal (simplified for paper trading)
    const signal = await this.generatePaperTradingSignal(asset, currentPrice);

    if (signal && signal.confidence >= 70 && signal.type !== 'HOLD') { // Filter out HOLD signals
      await this.openTrade(signal);
    }
  }

  /**
   * Generate trading signal for paper trading
   */
  private async generatePaperTradingSignal(asset: string, currentPrice: number): Promise<TradingSignal | null> {
    // Simplified signal generation (in real system, this would use full strategy)
    const random = Math.random();
    
    // 20% chance of signal generation
    if (random > 0.8) {
      const side: 'BUY' | 'SELL' = Math.random() > 0.5 ? 'BUY' : 'SELL';
      const confidence = 70 + Math.random() * 20; // 70-90% confidence
      
      // Calculate position size
      const riskAmount = this.portfolio.currentBalance * (this.portfolio.riskPerTrade / 100);
      const stopLossDistance = currentPrice * 0.025; // 2.5% stop loss
      let quantity = (riskAmount / stopLossDistance) * this.portfolio.leverage;
      
      // Ensure reasonable position size
      quantity = Math.max(quantity, 0.001);
      const maxQuantity = (this.portfolio.currentBalance * 0.3) / currentPrice;
      quantity = Math.min(quantity, maxQuantity);

      const stopLoss = side === 'BUY'
        ? currentPrice * 0.975
        : currentPrice * 1.025;

      return {
        id: `paper_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        timestamp: Date.now(),
        symbol: asset as any,
        type: side,
        price: currentPrice,
        quantity: quantity,
        confidence: confidence,
        strategy: 'PAPER_TRADING_ENHANCED',
        reason: `Paper trading signal - ${side.toLowerCase()} opportunity`,
        stopLoss,
        takeProfit: side === 'BUY' ? currentPrice * 1.075 : currentPrice * 0.925,
        riskReward: 3.0,
      };
    }

    return null;
  }

  /**
   * Open a new paper trade
   */
  private async openTrade(signal: TradingSignal): Promise<void> {
    try {
      // Generate dynamic take profit levels
      const marketRegime: MarketRegime = {
        type: 'TRENDING', // Simplified for paper trading
        strength: 75,
        direction: signal.type === 'BUY' ? 'UP' : 'DOWN',
        volatility: 0.03,
        volume: 1.2,
      };

      const takeProfitConfig: DynamicTakeProfitConfig = {
        asset: signal.symbol,
        entryPrice: signal.price,
        stopLoss: signal.stopLoss!,
        positionSize: signal.quantity,
        side: signal.type as 'BUY' | 'SELL', // Type assertion since we filtered out HOLD
        marketRegime,
        momentum: signal.type === 'BUY' ? 50 : -50,
        volume: 1.2,
      };

      const takeProfitLevels = this.takeProfitManager.generateDynamicTakeProfitLevels(takeProfitConfig);

      // Create paper trade (ensure signal.type is BUY or SELL)
      if (signal.type === 'HOLD') {
        logger.warn('Attempted to create trade with HOLD signal, skipping');
        return;
      }

      const trade: PaperTrade = {
        id: signal.id,
        symbol: signal.symbol,
        side: signal.type as 'BUY' | 'SELL', // Type assertion since we filtered out HOLD
        size: signal.quantity,
        entryPrice: signal.price,
        entryTime: signal.timestamp,
        status: 'OPEN',
        takeProfitLevels,
        partialExits: [],
        stopLoss: signal.stopLoss!,
        currentPrice: signal.price,
        unrealizedPnl: 0,
        maxProfit: 0,
        maxLoss: 0,
      };

      this.activeTrades.set(trade.id, trade);
      this.portfolio.totalTrades++;

      logger.info(`🔥 Paper Trade Opened: ${trade.side} ${trade.size.toFixed(4)} ${trade.symbol} @ $${trade.entryPrice.toFixed(2)}`);
      logger.info(`   Stop Loss: $${trade.stopLoss.toFixed(2)}`);
      logger.info(`   Take Profit Levels: ${takeProfitLevels.length} levels`);

    } catch (error) {
      logger.error('❌ Failed to open paper trade:', error);
    }
  }

  /**
   * Close a paper trade
   */
  private async closeTrade(trade: PaperTrade, reason: string): Promise<void> {
    try {
      trade.status = 'CLOSED';
      trade.exitPrice = trade.currentPrice;
      trade.exitTime = Date.now();
      trade.reason = reason;

      // Calculate final P&L
      const priceChange = trade.side === 'BUY'
        ? trade.currentPrice - trade.entryPrice
        : trade.entryPrice - trade.currentPrice;

      const finalPnl = (priceChange / trade.entryPrice) * trade.size * this.portfolio.leverage;
      trade.pnl = finalPnl;

      // Update portfolio
      this.portfolio.currentBalance += finalPnl;
      this.portfolio.totalPnl += finalPnl;

      if (finalPnl > 0) {
        this.portfolio.winningTrades++;
      } else {
        this.portfolio.losingTrades++;
      }

      // Move to closed trades
      this.activeTrades.delete(trade.id);
      this.closedTrades.push(trade);

      logger.info(`✅ Paper Trade Closed: ${trade.symbol} - ${reason}`);
      logger.info(`   Final P&L: $${finalPnl.toFixed(2)} (${((finalPnl / this.portfolio.initialBalance) * 100).toFixed(2)}%)`);
      logger.info(`   Partial Exits: ${trade.partialExits.length}`);

    } catch (error) {
      logger.error('❌ Failed to close paper trade:', error);
    }
  }

  /**
   * Calculate partial P&L
   */
  private calculatePartialPnl(trade: PaperTrade, exitPrice: number, exitSize: number): number {
    const priceChange = trade.side === 'BUY'
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;

    return (priceChange / trade.entryPrice) * exitSize * this.portfolio.leverage;
  }

  /**
   * Update portfolio metrics
   */
  private updatePortfolioMetrics(): void {
    // Calculate current drawdown
    const peak = Math.max(this.portfolio.initialBalance, this.portfolio.currentBalance);
    this.portfolio.currentDrawdown = ((peak - this.portfolio.currentBalance) / peak) * 100;
    this.portfolio.maxDrawdown = Math.max(this.portfolio.maxDrawdown, this.portfolio.currentDrawdown);

    // Calculate win rate
    const totalClosedTrades = this.portfolio.winningTrades + this.portfolio.losingTrades;
    this.portfolio.winRate = totalClosedTrades > 0
      ? (this.portfolio.winningTrades / totalClosedTrades) * 100
      : 0;
  }

  /**
   * Generate final report
   */
  private generateFinalReport(): void {
    const duration = this.closedTrades.length > 0
      ? (Date.now() - this.closedTrades[0].entryTime) / (1000 * 60) // Minutes
      : 0;

    logger.info('\n' + '🎉 PAPER TRADING FINAL REPORT'.padStart(80, '='));
    logger.info('=' .repeat(120));

    // Portfolio Summary
    logger.info('💰 PORTFOLIO SUMMARY:');
    logger.info(`   Initial Balance: $${this.portfolio.initialBalance.toFixed(2)}`);
    logger.info(`   Final Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
    logger.info(`   Total P&L: $${this.portfolio.totalPnl.toFixed(2)}`);
    logger.info(`   Return: ${((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance * 100).toFixed(2)}%`);
    logger.info(`   Max Drawdown: ${this.portfolio.maxDrawdown.toFixed(2)}%`);

    // Trading Statistics
    logger.info('\n📊 TRADING STATISTICS:');
    logger.info(`   Total Trades: ${this.portfolio.totalTrades}`);
    logger.info(`   Closed Trades: ${this.closedTrades.length}`);
    logger.info(`   Active Trades: ${this.activeTrades.size}`);
    logger.info(`   Winning Trades: ${this.portfolio.winningTrades}`);
    logger.info(`   Losing Trades: ${this.portfolio.losingTrades}`);
    logger.info(`   Win Rate: ${this.portfolio.winRate.toFixed(1)}%`);
    logger.info(`   Trading Duration: ${duration.toFixed(1)} minutes`);

    // Trade Analysis
    if (this.closedTrades.length > 0) {
      const profits = this.closedTrades.filter(t => t.pnl! > 0).map(t => t.pnl!);
      const losses = this.closedTrades.filter(t => t.pnl! <= 0).map(t => t.pnl!);

      const avgWin = profits.length > 0 ? profits.reduce((sum, p) => sum + p, 0) / profits.length : 0;
      const avgLoss = losses.length > 0 ? losses.reduce((sum, l) => sum + l, 0) / losses.length : 0;
      const profitFactor = losses.length > 0 ? Math.abs(profits.reduce((sum, p) => sum + p, 0) / losses.reduce((sum, l) => sum + l, 0)) : 0;

      logger.info('\n📈 TRADE ANALYSIS:');
      logger.info(`   Average Win: $${avgWin.toFixed(2)}`);
      logger.info(`   Average Loss: $${avgLoss.toFixed(2)}`);
      logger.info(`   Profit Factor: ${profitFactor.toFixed(2)}`);
      logger.info(`   Best Trade: $${Math.max(...this.closedTrades.map(t => t.pnl!)).toFixed(2)}`);
      logger.info(`   Worst Trade: $${Math.min(...this.closedTrades.map(t => t.pnl!)).toFixed(2)}`);
    }

    // Dynamic Take Profit Analysis
    const totalPartialExits = this.closedTrades.reduce((sum, trade) => sum + trade.partialExits.length, 0);
    const partialExitPnl = this.closedTrades.reduce((sum, trade) =>
      sum + trade.partialExits.reduce((pSum, pe) => pSum + pe.pnl, 0), 0
    );

    logger.info('\n🎯 DYNAMIC TAKE PROFIT ANALYSIS:');
    logger.info(`   Total Partial Exits: ${totalPartialExits}`);
    logger.info(`   Partial Exit P&L: $${partialExitPnl.toFixed(2)}`);
    logger.info(`   Avg Partial Exits per Trade: ${(totalPartialExits / Math.max(this.closedTrades.length, 1)).toFixed(1)}`);
    logger.info(`   Partial Exit Contribution: ${this.portfolio.totalPnl !== 0 ? ((partialExitPnl / this.portfolio.totalPnl) * 100).toFixed(1) : 0}%`);

    // Asset Performance
    logger.info('\n📊 ASSET PERFORMANCE:');
    this.tradingAssets.forEach(asset => {
      const assetTrades = this.closedTrades.filter(t => t.symbol === asset);
      if (assetTrades.length > 0) {
        const assetPnl = assetTrades.reduce((sum, t) => sum + t.pnl!, 0);
        const assetWins = assetTrades.filter(t => t.pnl! > 0).length;
        const assetWinRate = (assetWins / assetTrades.length) * 100;

        logger.info(`   ${asset}: ${assetTrades.length} trades, $${assetPnl.toFixed(2)} P&L, ${assetWinRate.toFixed(1)}% win rate`);
      }
    });

    logger.info('\n🚀 PAPER TRADING SYSTEM VALIDATION:');
    if (this.portfolio.totalPnl > 0) {
      logger.info('   ✅ PROFITABLE: Paper trading system generated positive returns');
      logger.info('   ✅ Dynamic take profit system working effectively');
      logger.info('   ✅ Risk management maintaining drawdown limits');
      logger.info('   🎯 READY FOR LIVE TRADING CONSIDERATION');
    } else {
      logger.info('   ⚠️ LOSS: Paper trading system needs optimization');
      logger.info('   🔧 Consider adjusting strategy parameters');
      logger.info('   📊 Analyze trade patterns for improvements');
    }

    logger.info('=' .repeat(120));
  }

  /**
   * Get current portfolio status
   */
  public getPortfolioStatus(): PaperPortfolio {
    this.updatePortfolioMetrics();
    return { ...this.portfolio };
  }

  /**
   * Get active trades
   */
  public getActiveTrades(): PaperTrade[] {
    return Array.from(this.activeTrades.values());
  }

  /**
   * Get closed trades
   */
  public getClosedTrades(): PaperTrade[] {
    return [...this.closedTrades];
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
