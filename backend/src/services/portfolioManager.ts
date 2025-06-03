/**
 * Portfolio Manager for Backtesting
 * Manages positions, trades, and portfolio performance
 */

import { 
  Position, 
  Trade, 
  PortfolioSnapshot, 
  TradingSignal, 
  BacktestConfig 
} from '../types/marketData';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';

export class PortfolioManager {
  private config: BacktestConfig;
  private cash: number;
  private positions: Map<string, Position> = new Map();
  private trades: Trade[] = [];
  private portfolioHistory: PortfolioSnapshot[] = [];
  private highWaterMark: number;
  private maxDrawdown: number = 0;
  private totalCommission: number = 0;

  constructor(config: BacktestConfig) {
    this.config = config;
    this.cash = config.initialCapital;
    this.highWaterMark = config.initialCapital;
    
    logger.info('💼 Portfolio Manager initialized', {
      initialCapital: config.initialCapital,
      leverage: config.leverage,
      riskPerTrade: config.riskPerTrade,
      commission: config.commission,
    });
  }

  /**
   * Execute a trading signal
   */
  public executeTrade(signal: TradingSignal, currentPrice: number, timestamp: number): Trade | null {
    try {
      logger.debug(`🔄 Attempting to execute trade:`, {
        symbol: signal.symbol,
        type: signal.type,
        price: currentPrice,
        quantity: signal.quantity,
        confidence: signal.confidence,
        cash: this.cash,
      });

      if (signal.type === 'BUY') {
        return this.openLongPosition(signal, currentPrice, timestamp);
      } else if (signal.type === 'SELL') {
        return this.openShortPosition(signal, currentPrice, timestamp);
      }

      logger.debug(`⚠️ Signal type ${signal.type} not handled`);
      return null;
    } catch (error) {
      logger.error('❌ Failed to execute trade:', error);
      return null;
    }
  }

  /**
   * Open a long position
   */
  private openLongPosition(signal: TradingSignal, currentPrice: number, timestamp: number): Trade | null {
    const positionKey = `${signal.symbol}_LONG`;
    
    // Check if we already have a long position
    if (this.positions.has(positionKey)) {
      logger.debug(`📊 Already have long position for ${signal.symbol}`);
      return null;
    }

    // Close any existing short position first
    const shortPositionKey = `${signal.symbol}_SHORT`;
    if (this.positions.has(shortPositionKey)) {
      this.closePosition(shortPositionKey, currentPrice, timestamp, 'Position reversal');
    }

    // Calculate position size and cost
    const quantity = this.calculatePositionSize(signal, currentPrice);
    const cost = quantity * currentPrice;
    const commission = cost * (this.config.commission / 100);
    const totalCost = cost + commission;

    logger.debug(`📊 Position calculation:`, {
      signalQuantity: signal.quantity,
      calculatedQuantity: quantity,
      cost,
      commission,
      totalCost,
      leverage: this.config.leverage,
    });

    // Check if we have enough cash (considering leverage)
    const requiredCash = totalCost / this.config.leverage;
    logger.debug(`💰 Cash check:`, {
      requiredCash,
      availableCash: this.cash,
      sufficient: requiredCash <= this.cash,
    });

    if (requiredCash > this.cash) {
      logger.info(`💰 INSUFFICIENT CASH: Required: $${requiredCash.toFixed(2)}, Available: $${this.cash.toFixed(2)}`);
      return null;
    }

    // Create position
    const position: Position = {
      symbol: signal.symbol,
      side: 'LONG',
      size: quantity,
      entryPrice: currentPrice,
      entryTime: timestamp,
      currentPrice: currentPrice,
      unrealizedPnl: 0,
      leverage: this.config.leverage,
    };

    // Update cash and positions
    this.cash -= requiredCash;
    this.totalCommission += commission;
    this.positions.set(positionKey, position);

    logger.info(`📈 Opened LONG position`, {
      symbol: signal.symbol,
      quantity,
      price: currentPrice,
      cost: totalCost,
      commission,
      remainingCash: this.cash,
    });

    // Return trade record (entry only, will be completed when closed)
    return {
      id: createEventId(),
      symbol: signal.symbol,
      side: 'LONG',
      entryPrice: currentPrice,
      exitPrice: 0, // Will be set when position is closed
      quantity,
      entryTime: timestamp,
      exitTime: 0, // Will be set when position is closed
      pnl: 0, // Will be calculated when position is closed
      pnlPercent: 0,
      commission,
      strategy: signal.strategy,
      reason: signal.reason,
      duration: 0,
    };
  }

  /**
   * Open a short position
   */
  private openShortPosition(signal: TradingSignal, currentPrice: number, timestamp: number): Trade | null {
    const positionKey = `${signal.symbol}_SHORT`;
    
    // Check if we already have a short position
    if (this.positions.has(positionKey)) {
      logger.debug(`📊 Already have short position for ${signal.symbol}`);
      return null;
    }

    // Close any existing long position first
    const longPositionKey = `${signal.symbol}_LONG`;
    if (this.positions.has(longPositionKey)) {
      this.closePosition(longPositionKey, currentPrice, timestamp, 'Position reversal');
    }

    // Calculate position size and cost
    const quantity = this.calculatePositionSize(signal, currentPrice);
    const cost = quantity * currentPrice;
    const commission = cost * (this.config.commission / 100);
    const totalCost = cost + commission;

    // Check if we have enough cash (considering leverage)
    const requiredCash = totalCost / this.config.leverage;
    if (requiredCash > this.cash) {
      logger.info(`💰 INSUFFICIENT CASH (SHORT): Required: $${requiredCash.toFixed(2)}, Available: $${this.cash.toFixed(2)}`);
      return null;
    }

    // Create position
    const position: Position = {
      symbol: signal.symbol,
      side: 'SHORT',
      size: quantity,
      entryPrice: currentPrice,
      entryTime: timestamp,
      currentPrice: currentPrice,
      unrealizedPnl: 0,
      leverage: this.config.leverage,
    };

    // Update cash and positions
    this.cash -= requiredCash;
    this.totalCommission += commission;
    this.positions.set(positionKey, position);

    logger.info(`📉 Opened SHORT position`, {
      symbol: signal.symbol,
      quantity,
      price: currentPrice,
      cost: totalCost,
      commission,
      remainingCash: this.cash,
    });

    // Return trade record (entry only, will be completed when closed)
    return {
      id: createEventId(),
      symbol: signal.symbol,
      side: 'SHORT',
      entryPrice: currentPrice,
      exitPrice: 0,
      quantity,
      entryTime: timestamp,
      exitTime: 0,
      pnl: 0,
      pnlPercent: 0,
      commission,
      strategy: signal.strategy,
      reason: signal.reason,
      duration: 0,
    };
  }

  /**
   * Close a position
   */
  public closePosition(positionKey: string, exitPrice: number, timestamp: number, reason: string): Trade | null {
    const position = this.positions.get(positionKey);
    if (!position) {
      return null;
    }

    // Calculate P&L
    const pnl = this.calculatePnL(position, exitPrice);
    const pnlPercent = (pnl / (position.size * position.entryPrice)) * 100;
    
    // Calculate exit commission
    const exitCost = position.size * exitPrice;
    const exitCommission = exitCost * (this.config.commission / 100);
    const totalCommission = exitCommission; // Entry commission already deducted
    
    // Net P&L after commission
    const netPnl = pnl - totalCommission;
    
    // Update cash
    const returnedCash = (position.size * position.entryPrice) / this.config.leverage + netPnl;
    this.cash += returnedCash;
    this.totalCommission += exitCommission;

    // Create completed trade record
    const trade: Trade = {
      id: createEventId(),
      symbol: position.symbol,
      side: position.side,
      entryPrice: position.entryPrice,
      exitPrice,
      quantity: position.size,
      entryTime: position.entryTime,
      exitTime: timestamp,
      pnl: netPnl,
      pnlPercent,
      commission: totalCommission,
      strategy: 'Unknown', // Will be updated by caller
      reason,
      duration: timestamp - position.entryTime,
    };

    // Remove position
    this.positions.delete(positionKey);
    this.trades.push(trade);

    logger.info(`💰 Closed ${position.side} position`, {
      symbol: position.symbol,
      entryPrice: position.entryPrice,
      exitPrice,
      pnl: netPnl,
      pnlPercent: pnlPercent.toFixed(2),
      duration: this.formatDuration(trade.duration),
      reason,
    });

    return trade;
  }

  /**
   * Update all positions with current market prices
   */
  public updatePositions(symbol: string, currentPrice: number, timestamp: number): void {
    for (const [key, position] of this.positions) {
      if (position.symbol === symbol) {
        position.currentPrice = currentPrice;
        position.unrealizedPnl = this.calculatePnL(position, currentPrice);
      }
    }
  }

  /**
   * Check for stop loss and take profit triggers
   */
  public checkStopLossAndTakeProfit(symbol: string, currentPrice: number, timestamp: number, signal?: TradingSignal): Trade[] {
    const closedTrades: Trade[] = [];
    
    for (const [key, position] of this.positions) {
      if (position.symbol !== symbol) continue;

      let shouldClose = false;
      let reason = '';

      // Check stop loss and take profit if signal provided
      if (signal?.stopLoss && signal?.takeProfit) {
        if (position.side === 'LONG') {
          if (currentPrice <= signal.stopLoss) {
            shouldClose = true;
            reason = 'Stop loss triggered';
          } else if (currentPrice >= signal.takeProfit) {
            shouldClose = true;
            reason = 'Take profit triggered';
          }
        } else { // SHORT
          if (currentPrice >= signal.stopLoss) {
            shouldClose = true;
            reason = 'Stop loss triggered';
          } else if (currentPrice <= signal.takeProfit) {
            shouldClose = true;
            reason = 'Take profit triggered';
          }
        }
      }

      if (shouldClose) {
        const trade = this.closePosition(key, currentPrice, timestamp, reason);
        if (trade) {
          closedTrades.push(trade);
        }
      }
    }

    return closedTrades;
  }

  /**
   * Create portfolio snapshot
   */
  public createSnapshot(timestamp: number): PortfolioSnapshot {
    const totalValue = this.getTotalPortfolioValue();
    const totalPnl = totalValue - this.config.initialCapital;
    const totalPnlPercent = (totalPnl / this.config.initialCapital) * 100;
    
    // Calculate drawdown
    if (totalValue > this.highWaterMark) {
      this.highWaterMark = totalValue;
    }
    
    const currentDrawdown = (this.highWaterMark - totalValue) / this.highWaterMark * 100;
    if (currentDrawdown > this.maxDrawdown) {
      this.maxDrawdown = currentDrawdown;
    }

    const snapshot: PortfolioSnapshot = {
      timestamp,
      totalValue,
      cash: this.cash,
      positions: Array.from(this.positions.values()),
      totalPnl,
      totalPnlPercent,
      drawdown: currentDrawdown,
      maxDrawdown: this.maxDrawdown,
      leverage: this.config.leverage,
    };

    this.portfolioHistory.push(snapshot);
    return snapshot;
  }

  /**
   * Calculate position size based on signal and risk management
   */
  private calculatePositionSize(signal: TradingSignal, currentPrice: number): number {
    // Use the quantity from the signal if provided, otherwise calculate
    if (signal.quantity && signal.quantity > 0) {
      return signal.quantity;
    }

    // Calculate based on risk per trade
    const riskAmount = this.config.initialCapital * (this.config.riskPerTrade / 100);
    const stopLossDistance = signal.stopLoss ? Math.abs(currentPrice - signal.stopLoss) : currentPrice * 0.02;
    
    let positionSize = riskAmount / stopLossDistance;
    positionSize *= this.config.leverage;
    
    return Math.max(positionSize, 0.001);
  }

  /**
   * Calculate P&L for a position
   */
  private calculatePnL(position: Position, currentPrice: number): number {
    if (position.side === 'LONG') {
      return (currentPrice - position.entryPrice) * position.size;
    } else {
      return (position.entryPrice - currentPrice) * position.size;
    }
  }

  /**
   * Get total portfolio value
   */
  private getTotalPortfolioValue(): number {
    let totalValue = this.cash;
    
    for (const position of this.positions.values()) {
      const positionValue = position.size * position.currentPrice;
      const unrealizedPnl = this.calculatePnL(position, position.currentPrice);
      totalValue += (positionValue / this.config.leverage) + unrealizedPnl;
    }
    
    return totalValue;
  }

  /**
   * Format duration in human readable format
   */
  private formatDuration(durationMs: number): string {
    const hours = Math.floor(durationMs / (1000 * 60 * 60));
    const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}h ${minutes}m`;
  }

  // Getters
  public getCash(): number { return this.cash; }
  public getPositions(): Position[] { return Array.from(this.positions.values()); }
  public getTrades(): Trade[] { return [...this.trades]; }
  public getPortfolioHistory(): PortfolioSnapshot[] { return [...this.portfolioHistory]; }
  public getMaxDrawdown(): number { return this.maxDrawdown; }
  public getTotalCommission(): number { return this.totalCommission; }
}
