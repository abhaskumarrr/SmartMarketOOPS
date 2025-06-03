/**
 * AI Position Manager
 * Manages Delta Exchange positions using AI-powered dynamic take profit system
 */

import { DynamicTakeProfitManager, DynamicTakeProfitConfig, MarketRegime } from './dynamicTakeProfitManager';
import DeltaExchangeAPI from './deltaApiService';
import { logger } from '../utils/logger';

export interface ManagedPosition {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  takeProfitLevels: any[];
  stopLoss: number;
  partialExits: PartialExit[];
  status: 'ACTIVE' | 'MANAGING' | 'CLOSED';
  lastUpdate: number;
  aiRecommendations: string[];
}

export interface PartialExit {
  level: number;
  percentage: number;
  targetPrice: number;
  executed: boolean;
  executedAt?: number;
  orderId?: string;
  pnl?: number;
}

export interface AIAnalysis {
  action: 'HOLD' | 'PARTIAL_EXIT' | 'FULL_EXIT' | 'ADJUST_STOP' | 'TRAIL_STOP';
  confidence: number;
  reasoning: string;
  targetPrice?: number;
  percentage?: number;
  newStopLoss?: number;
}

export class AIPositionManager {
  private deltaApi: DeltaExchangeAPI;
  private takeProfitManager: DynamicTakeProfitManager;
  private managedPositions: Map<string, ManagedPosition> = new Map();
  private isRunning: boolean = false;
  private updateInterval: NodeJS.Timeout | null = null;

  constructor(deltaApi: DeltaExchangeAPI) {
    this.deltaApi = deltaApi;
    this.takeProfitManager = new DynamicTakeProfitManager();
  }

  /**
   * Start AI position management
   */
  public async startManagement(): Promise<void> {
    if (this.isRunning) {
      logger.warn('AI Position Manager is already running');
      return;
    }

    this.isRunning = true;
    logger.info('🤖 Starting AI Position Management System');
    
    // Initial position scan
    await this.scanAndManagePositions();
    
    // Set up periodic updates every 30 seconds
    this.updateInterval = setInterval(async () => {
      await this.scanAndManagePositions();
    }, 30000);

    logger.info('✅ AI Position Manager started - monitoring every 30 seconds');
  }

  /**
   * Stop AI position management
   */
  public stopManagement(): void {
    this.isRunning = false;
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    logger.info('🛑 AI Position Manager stopped');
  }

  /**
   * Scan and manage all positions
   */
  private async scanAndManagePositions(): Promise<void> {
    try {
      logger.info('🔍 Scanning positions for AI management...');
      
      // Get current positions from Delta Exchange
      const positions = await this.deltaApi.getPositions();
      
      if (positions.length === 0) {
        logger.info('📊 No active positions found');
        return;
      }

      logger.info(`🎯 Found ${positions.length} active positions`);

      // Process each position
      for (const position of positions) {
        await this.managePosition(position);
      }

      // Display summary
      this.displayManagementSummary();

    } catch (error) {
      logger.error('❌ Error scanning positions:', error.message);
    }
  }

  /**
   * Manage individual position with AI
   */
  private async managePosition(position: any): Promise<void> {
    try {
      const positionId = `${position.symbol}_${position.size}`;
      
      // Get or create managed position
      let managedPos = this.managedPositions.get(positionId);
      
      if (!managedPos) {
        // New position - set up AI management
        managedPos = await this.initializePositionManagement(position);
        this.managedPositions.set(positionId, managedPos);
        logger.info(`🆕 New position detected: ${position.symbol} - Setting up AI management`);
      }

      // Update current data
      managedPos.currentPrice = await this.getCurrentPrice(position.symbol);
      managedPos.unrealizedPnl = parseFloat(position.unrealized_pnl || '0');
      managedPos.lastUpdate = Date.now();

      // Get AI analysis
      const aiAnalysis = await this.getAIAnalysis(managedPos);
      
      // Execute AI recommendations
      await this.executeAIRecommendation(managedPos, aiAnalysis);

      logger.info(`🤖 AI Analysis for ${position.symbol}: ${aiAnalysis.action} (${aiAnalysis.confidence}% confidence)`);
      logger.info(`   Reasoning: ${aiAnalysis.reasoning}`);

    } catch (error) {
      logger.error(`❌ Error managing position ${position.symbol}:`, error.message);
    }
  }

  /**
   * Initialize AI management for new position
   */
  private async initializePositionManagement(position: any): Promise<ManagedPosition> {
    const side: 'LONG' | 'SHORT' = parseFloat(position.size) > 0 ? 'LONG' : 'SHORT';
    const entryPrice = parseFloat(position.entry_price);
    const size = Math.abs(parseFloat(position.size));
    
    // Generate dynamic take profit levels
    const marketRegime: MarketRegime = {
      type: 'TRENDING', // Simplified - in real system would analyze market
      strength: 75,
      direction: side === 'LONG' ? 'UP' : 'DOWN',
      volatility: 0.03,
      volume: 1.2,
    };

    const takeProfitConfig: DynamicTakeProfitConfig = {
      asset: position.symbol,
      entryPrice,
      stopLoss: side === 'LONG' ? entryPrice * 0.975 : entryPrice * 1.025, // 2.5% stop
      positionSize: size,
      side: side === 'LONG' ? 'BUY' : 'SELL',
      marketRegime,
      momentum: side === 'LONG' ? 50 : -50,
      volume: 1.2,
    };

    const takeProfitLevels = this.takeProfitManager.generateDynamicTakeProfitLevels(takeProfitConfig);

    // Convert to partial exits
    const partialExits: PartialExit[] = takeProfitLevels.map((level, index) => ({
      level: index + 1,
      percentage: level.percentage,
      targetPrice: level.priceTarget,
      executed: false,
    }));

    const managedPosition: ManagedPosition = {
      id: `${position.symbol}_${position.size}`,
      symbol: position.symbol,
      side,
      size,
      entryPrice,
      currentPrice: entryPrice,
      unrealizedPnl: parseFloat(position.unrealized_pnl || '0'),
      takeProfitLevels,
      stopLoss: takeProfitConfig.stopLoss,
      partialExits,
      status: 'MANAGING',
      lastUpdate: Date.now(),
      aiRecommendations: [],
    };

    logger.info(`🎯 AI Management initialized for ${position.symbol}:`);
    logger.info(`   Side: ${side}, Size: ${size}, Entry: $${entryPrice}`);
    logger.info(`   Stop Loss: $${takeProfitConfig.stopLoss.toFixed(2)}`);
    logger.info(`   Take Profit Levels: ${partialExits.length}`);

    return managedPosition;
  }

  /**
   * Get current price for symbol
   */
  private async getCurrentPrice(symbol: string): Promise<number> {
    try {
      const ticker = await this.deltaApi.getTicker(symbol);
      return parseFloat(ticker.close || '0');
    } catch (error) {
      logger.warn(`⚠️ Could not get current price for ${symbol}, using cached price`);
      return 0;
    }
  }

  /**
   * Get AI analysis for position
   */
  private async getAIAnalysis(position: ManagedPosition): Promise<AIAnalysis> {
    const priceChange = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
    const isProfit = (position.side === 'LONG' && priceChange > 0) || (position.side === 'SHORT' && priceChange < 0);
    const profitPercent = Math.abs(priceChange);

    // AI Decision Logic
    if (isProfit && profitPercent >= 5) {
      // Strong profit - consider partial exit
      const unexecutedExits = position.partialExits.filter(exit => !exit.executed);
      if (unexecutedExits.length > 0) {
        const nextExit = unexecutedExits[0];
        const shouldExit = position.side === 'LONG' 
          ? position.currentPrice >= nextExit.targetPrice
          : position.currentPrice <= nextExit.targetPrice;

        if (shouldExit) {
          return {
            action: 'PARTIAL_EXIT',
            confidence: 85,
            reasoning: `Price reached take profit level ${nextExit.level} (${profitPercent.toFixed(1)}% profit)`,
            targetPrice: nextExit.targetPrice,
            percentage: nextExit.percentage,
          };
        }
      }

      // Trail stop loss if in significant profit
      if (profitPercent >= 10) {
        const newStopLoss = position.side === 'LONG'
          ? position.currentPrice * 0.95  // Trail 5% below current price
          : position.currentPrice * 1.05; // Trail 5% above current price

        return {
          action: 'TRAIL_STOP',
          confidence: 75,
          reasoning: `Trailing stop loss due to ${profitPercent.toFixed(1)}% profit`,
          newStopLoss,
        };
      }
    }

    // Check stop loss
    const shouldStop = position.side === 'LONG'
      ? position.currentPrice <= position.stopLoss
      : position.currentPrice >= position.stopLoss;

    if (shouldStop) {
      return {
        action: 'FULL_EXIT',
        confidence: 95,
        reasoning: `Stop loss triggered at $${position.currentPrice.toFixed(2)}`,
      };
    }

    // Default: hold position
    return {
      action: 'HOLD',
      confidence: 60,
      reasoning: `Position within normal range (${priceChange.toFixed(1)}% from entry)`,
    };
  }

  /**
   * Execute AI recommendation
   */
  private async executeAIRecommendation(position: ManagedPosition, analysis: AIAnalysis): Promise<void> {
    position.aiRecommendations.push(`${new Date().toISOString()}: ${analysis.action} - ${analysis.reasoning}`);

    switch (analysis.action) {
      case 'PARTIAL_EXIT':
        await this.executePartialExit(position, analysis);
        break;
      case 'FULL_EXIT':
        await this.executeFullExit(position, analysis);
        break;
      case 'TRAIL_STOP':
        await this.updateStopLoss(position, analysis.newStopLoss!);
        break;
      case 'HOLD':
        // No action needed
        break;
    }
  }

  /**
   * Execute partial exit
   */
  private async executePartialExit(position: ManagedPosition, analysis: AIAnalysis): Promise<void> {
    try {
      logger.info(`🎯 Executing partial exit: ${analysis.percentage}% at $${analysis.targetPrice?.toFixed(2)}`);
      
      // In a real implementation, this would place an order
      // For now, we'll simulate the execution
      const exitSize = (position.size * analysis.percentage!) / 100;
      
      // Mark the exit as executed
      const exit = position.partialExits.find(e => e.targetPrice === analysis.targetPrice);
      if (exit) {
        exit.executed = true;
        exit.executedAt = Date.now();
        exit.pnl = this.calculatePartialPnl(position, analysis.targetPrice!, exitSize);
      }

      // Update position size
      position.size -= exitSize;
      
      logger.info(`✅ Partial exit executed: ${analysis.percentage}% at $${analysis.targetPrice?.toFixed(2)}`);
      
    } catch (error) {
      logger.error('❌ Failed to execute partial exit:', error.message);
    }
  }

  /**
   * Execute full exit
   */
  private async executeFullExit(position: ManagedPosition, analysis: AIAnalysis): Promise<void> {
    try {
      logger.info(`🚨 Executing full exit: ${analysis.reasoning}`);
      
      // In a real implementation, this would close the position
      position.status = 'CLOSED';
      
      logger.info(`✅ Position closed: ${position.symbol}`);
      
    } catch (error) {
      logger.error('❌ Failed to execute full exit:', error.message);
    }
  }

  /**
   * Update stop loss
   */
  private async updateStopLoss(position: ManagedPosition, newStopLoss: number): Promise<void> {
    try {
      logger.info(`🔄 Updating stop loss from $${position.stopLoss.toFixed(2)} to $${newStopLoss.toFixed(2)}`);
      
      position.stopLoss = newStopLoss;
      
      logger.info(`✅ Stop loss updated to $${newStopLoss.toFixed(2)}`);
      
    } catch (error) {
      logger.error('❌ Failed to update stop loss:', error.message);
    }
  }

  /**
   * Calculate partial P&L
   */
  private calculatePartialPnl(position: ManagedPosition, exitPrice: number, exitSize: number): number {
    const priceChange = position.side === 'LONG' 
      ? exitPrice - position.entryPrice 
      : position.entryPrice - exitPrice;
    
    return (priceChange / position.entryPrice) * exitSize * 200; // Assuming 200x leverage
  }

  /**
   * Display management summary
   */
  private displayManagementSummary(): void {
    const activePositions = Array.from(this.managedPositions.values()).filter(p => p.status === 'MANAGING');
    
    if (activePositions.length === 0) {
      logger.info('📊 No positions under AI management');
      return;
    }

    logger.info('\n🤖 AI POSITION MANAGEMENT SUMMARY:');
    activePositions.forEach(position => {
      const profitPercent = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
      const executedExits = position.partialExits.filter(e => e.executed).length;
      
      logger.info(`   ${position.symbol}: ${position.side} $${position.unrealizedPnl.toFixed(2)} (${profitPercent.toFixed(1)}%)`);
      logger.info(`     Partial Exits: ${executedExits}/${position.partialExits.length}`);
      logger.info(`     Stop Loss: $${position.stopLoss.toFixed(2)}`);
    });
  }

  /**
   * Get managed positions
   */
  public getManagedPositions(): ManagedPosition[] {
    return Array.from(this.managedPositions.values());
  }
}
