/**
 * Risk Assessment Service
 * Handles portfolio risk analysis, monitoring, and alerts
 */

import { v4 as uuidv4 } from 'uuid';
import prisma from '../../utils/prismaClient';
import { createLogger, LogData } from '../../utils/logger';
import { 
  RiskLevel,
  RiskAlertType,
  TradeRiskAnalysis,
  RiskReport,
  RiskAlert,
  PortfolioRiskMetrics
} from '../../types/risk';

// Create logger
const logger = createLogger('RiskAssessmentService');

/**
 * Risk Assessment Service class
 * Provides methods for analyzing and monitoring trading risk
 */
export class RiskAssessmentService {
  // Keep track of equity high watermark for drawdown calculations
  private equityHighWatermark: Record<string, number> = {};

  /**
   * Creates a new Risk Assessment Service instance
   */
  constructor() {
    logger.info('Risk Assessment Service initialized');
  }

  /**
   * Generate a comprehensive risk report for a user
   * @param userId - User ID
   * @returns Risk report
   */
  async generateRiskReport(userId: string): Promise<RiskReport> {
    try {
      logger.info(`Generating risk report for user ${userId}`);

      // Get account balance and equity
      const accountBalance = await this._getAccountBalance(userId);
      const accountEquity = await this._getAccountEquity(userId);
      
      // Get open positions
      const openPositions = await this._getOpenPositions(userId);
      
      // Calculate metrics
      const totalMargin = this._calculateTotalMargin(openPositions);
      const freeMargin = accountBalance - totalMargin;
      const marginLevel = totalMargin > 0 ? (accountEquity / totalMargin) * 100 : 100;
      
      // Calculate exposure
      const { exposureBySymbol, exposureByDirection, totalRisk } = this._calculateExposure(openPositions, accountBalance);
      
      // Calculate drawdown
      const { currentDrawdown, maxDrawdown } = this._calculateDrawdown(userId, accountEquity);
      
      // Calculate daily P&L
      const { dailyPnL, dailyPnLPercentage } = await this._calculateDailyPnL(userId, accountBalance);
      
      // Generate alerts
      const alerts = await this._generateAlerts(userId, {
        marginLevel,
        currentDrawdown,
        dailyPnLPercentage,
        exposureBySymbol,
        exposureByDirection,
        totalRisk
      });
      
      // Determine overall risk level
      const riskLevel = this._determineRiskLevel({
        marginLevel,
        currentDrawdown,
        dailyPnLPercentage,
        totalRisk
      });

      return {
        userId,
        timestamp: new Date().toISOString(),
        totalBalance: accountBalance,
        totalEquity: accountEquity,
        totalMargin,
        freeMargin,
        marginLevel,
        openPositions: openPositions.length,
        openPositionsRisk: totalRisk,
        maxDrawdown,
        currentDrawdown,
        dailyPnL,
        dailyPnLPercentage,
        exposureBySymbol,
        exposureByDirection,
        riskLevel,
        alerts
      };
    } catch (error) {
      const logData: LogData = {
        userId,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error generating risk report for user ${userId}`, logData);
      throw error;
    }
  }

  /**
   * Analyze risk for a specific trade/position
   * @param userId - User ID
   * @param positionId - Position ID
   * @returns Trade risk analysis
   */
  async analyzeTradeRisk(userId: string, positionId: string): Promise<TradeRiskAnalysis> {
    try {
      logger.info(`Analyzing trade risk for position ${positionId}`);

      // Get position
      const position = await prisma.position.findFirst({
        where: {
          id: positionId,
          userId
        }
      });

      if (!position) {
        throw new Error(`Position ${positionId} not found for user ${userId}`);
      }

      // Get account balance
      const accountBalance = await this._getAccountBalance(userId);
      
      // Calculate metrics
      const unrealizedPnL = position.currentPrice 
        ? (position.side.toLowerCase() === 'long'
          ? (position.currentPrice - position.entryPrice) * position.amount
          : (position.entryPrice - position.currentPrice) * position.amount)
        : 0;
      
      const unrealizedPnLPercentage = (unrealizedPnL / accountBalance) * 100;
      
      const distanceToStopLoss = position.stopLossPrice
        ? Math.abs((position.stopLossPrice - (position.currentPrice || position.entryPrice)) / (position.currentPrice || position.entryPrice)) * 100
        : 0;
        
      const distanceToTakeProfit = position.takeProfitPrice
        ? Math.abs((position.takeProfitPrice - (position.currentPrice || position.entryPrice)) / (position.currentPrice || position.entryPrice)) * 100
        : undefined;
      
      // Calculate breakeven price (including fees, assuming 0.1% fee)
      const feePercentage = 0.1;
      const feesAmount = (position.entryPrice * position.amount * feePercentage) / 100;
      const breakEvenPrice = position.side.toLowerCase() === 'long'
        ? position.entryPrice + (feesAmount / position.amount)
        : position.entryPrice - (feesAmount / position.amount);
        
      // Calculate risk-reward ratio
      const riskRewardRatio = (position.takeProfitPrice && position.stopLossPrice)
        ? Math.abs((position.takeProfitPrice - position.entryPrice) / (position.stopLossPrice - position.entryPrice))
        : 0;
        
      // Calculate time in trade
      const timeInTrade = Math.floor((Date.now() - position.openedAt.getTime()) / 1000);
      
      // Calculate exposure and risk
      const exposure = (position.amount * position.entryPrice) / accountBalance * 100;
      
      // Calculate risk amount based on stop loss
      const riskAmount = position.stopLossPrice
        ? Math.abs(position.stopLossPrice - position.entryPrice) * position.amount
        : position.amount * position.entryPrice; // If no stop loss, assume full position at risk
        
      const riskPercentage = (riskAmount / accountBalance) * 100;
      
      // Calculate margin and liquidation price (assuming leverage)
      const margin = (position.amount * position.entryPrice) / position.leverage;
      
      // Simple liquidation price calculation (can be refined based on exchange rules)
      const liquidationPrice = position.side.toLowerCase() === 'long'
        ? position.entryPrice * (1 - (1 / position.leverage) * 0.8) // 80% of margin used triggers liquidation
        : position.entryPrice * (1 + (1 / position.leverage) * 0.8);
        
      // Calculate normalized risk score (0-100)
      const riskFactors = [
        Math.min(riskPercentage * 5, 30), // Risk percentage contributes up to 30 points
        Math.min(exposure * 2, 20), // Exposure contributes up to 20 points
        position.stopLossPrice ? 0 : 20, // No stop loss adds 20 points
        position.leverage > 5 ? Math.min((position.leverage - 5) * 5, 20) : 0, // High leverage adds up to 20 points
        distanceToStopLoss < 1 ? 10 : 0 // Close to stop loss adds 10 points
      ];
      
      // Return analysis
      return {
        tradeId: position.id,
        symbol: position.symbol,
        direction: position.side.toLowerCase() as 'long' | 'short',
        entryPrice: position.entryPrice,
        currentPrice: position.currentPrice || position.entryPrice,
        positionSize: position.amount,
        exposure,
        riskAmount,
        riskPercentage,
        stopLossPrice: position.stopLossPrice || 0,
        takeProfitPrice: position.takeProfitPrice || undefined,
        unrealizedPnL,
        unrealizedPnLPercentage,
        distanceToStopLoss,
        distanceToTakeProfit,
        breakEvenPrice,
        riskRewardRatio,
        timeInTrade,
        margin,
        liquidationPrice,
        riskScore: Math.min(100, riskFactors.reduce((sum, value) => sum + value, 0)),
        riskLevel: this._determinePositionRiskLevel(riskFactors.reduce((sum, value) => sum + value, 0))
      };
    } catch (error) {
      const logData: LogData = {
        userId,
        positionId,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error analyzing trade risk for position ${positionId}`, logData);
      throw error;
    }
  }

  /**
   * Create a risk alert
   * @param userId - User ID
   * @param type - Alert type
   * @param level - Alert level
   * @param message - Alert message
   * @param details - Alert details
   * @returns Created alert
   */
  async createAlert(
    userId: string,
    type: RiskAlertType,
    level: 'info' | 'warning' | 'critical',
    message: string,
    details: Record<string, any>
  ): Promise<RiskAlert> {
    try {
      logger.info(`Creating ${level} risk alert for user ${userId}: ${type}`);
      
      const alert = await prisma.riskAlert.create({
        data: {
          id: uuidv4(),
          userId,
          type,
          level,
          message,
          details: details as any,
          timestamp: new Date(),
          acknowledged: false
        }
      });

      return {
        id: alert.id,
        userId: alert.userId,
        type: alert.type as RiskAlertType,
        level: alert.level as 'info' | 'warning' | 'critical',
        message: alert.message,
        details: alert.details as Record<string, any>,
        timestamp: alert.timestamp.toISOString(),
        acknowledged: alert.acknowledged,
        resolvedAt: alert.resolvedAt?.toISOString()
      };
    } catch (error) {
      const logData: LogData = {
        userId,
        type,
        level,
        message,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error creating risk alert for user ${userId}`, logData);
      throw error;
    }
  }

  /**
   * Get account balance for a user
   * @private
   * @param userId - User ID
   * @returns Account balance
   */
  private async _getAccountBalance(userId: string): Promise<number> {
    // In a real system, this would fetch from a balance service or exchange API
    // For now, return a placeholder balance of 10,000
    return 10000;
  }

  /**
   * Get account equity for a user
   * @private
   * @param userId - User ID
   * @returns Account equity
   */
  private async _getAccountEquity(userId: string): Promise<number> {
    // Get account balance
    const accountBalance = await this._getAccountBalance(userId);
    
    // Get open positions
    const openPositions = await this._getOpenPositions(userId);
    
    // Calculate unrealized P&L
    let unrealizedPnL = 0;
    
    for (const position of openPositions) {
      if (position.currentPrice) {
        const positionPnL = position.side.toLowerCase() === 'long'
          ? (position.currentPrice - position.entryPrice) * position.amount
          : (position.entryPrice - position.currentPrice) * position.amount;
          
        unrealizedPnL += positionPnL;
      }
    }
    
    // Return equity (balance + unrealized P&L)
    return accountBalance + unrealizedPnL;
  }

  /**
   * Get open positions for a user
   * @private
   * @param userId - User ID
   * @returns Open positions
   */
  private async _getOpenPositions(userId: string): Promise<any[]> {
    try {
      return await prisma.position.findMany({
        where: {
          userId,
          status: 'Open'
        }
      });
    } catch (error) {
      logger.error(`Error getting open positions for user ${userId}`, {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });
      
      return [];
    }
  }

  /**
   * Calculate total margin used
   * @private
   * @param positions - Open positions
   * @returns Total margin
   */
  private _calculateTotalMargin(positions: any[]): number {
    return positions.reduce((total, position) => {
      return total + ((position.amount * position.entryPrice) / position.leverage);
    }, 0);
  }

  /**
   * Calculate exposure by symbol and direction
   * @private
   * @param positions - Open positions
   * @param accountBalance - Account balance
   * @returns Exposure details
   */
  private _calculateExposure(positions: any[], accountBalance: number): any {
    const exposureBySymbol: Record<string, number> = {};
    const exposureByDirection: Record<string, number> = {};
    let totalRisk = 0;

    for (const position of positions) {
      // Calculate exposure for this position (position size / account balance)
      const positionExposure = (position.amount * position.entryPrice) / accountBalance * 100;
      
      // Calculate risk for this position
      const positionRisk = position.stopLossPrice
        ? Math.abs(position.stopLossPrice - position.entryPrice) * position.amount / accountBalance * 100
        : positionExposure; // If no stop loss, assume full position at risk
      
      // Add to symbol exposure
      exposureBySymbol[position.symbol] = (exposureBySymbol[position.symbol] || 0) + positionExposure;
      
      // Add to direction exposure
      const direction = position.side.toLowerCase();
      exposureByDirection[direction] = (exposureByDirection[direction] || 0) + positionExposure;
      
      // Add to total risk
      totalRisk += positionRisk;
    }

    return {
      exposureBySymbol,
      exposureByDirection,
      totalRisk
    };
  }

  /**
   * Calculate drawdown
   * @private
   * @param userId - User ID
   * @param currentEquity - Current equity
   * @returns Drawdown details
   */
  private _calculateDrawdown(userId: string, currentEquity: number): any {
    // Initialize high watermark if not exists
    if (!this.equityHighWatermark[userId]) {
      this.equityHighWatermark[userId] = currentEquity;
    }
    
    // Update high watermark if current equity is higher
    if (currentEquity > this.equityHighWatermark[userId]) {
      this.equityHighWatermark[userId] = currentEquity;
    }
    
    // Calculate current drawdown
    const currentDrawdown = this.equityHighWatermark[userId] > 0
      ? ((this.equityHighWatermark[userId] - currentEquity) / this.equityHighWatermark[userId]) * 100
      : 0;
    
    // For max drawdown, we should ideally store this in the database
    // For now, we'll just return current drawdown as max drawdown
    const maxDrawdown = currentDrawdown;

    return {
      currentDrawdown,
      maxDrawdown
    };
  }

  /**
   * Calculate daily P&L
   * @private
   * @param userId - User ID
   * @param accountBalance - Account balance
   * @returns Daily P&L details
   */
  private async _calculateDailyPnL(userId: string, accountBalance: number): Promise<any> {
    // Get today's closed positions
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const closedPositions = await prisma.position.findMany({
      where: {
        userId,
        status: 'Closed',
        closedAt: {
          gte: today
        }
      }
    });
    
    // Calculate realized P&L
    let dailyPnL = 0;
    
    for (const position of closedPositions) {
      if (position.pnl) {
        dailyPnL += position.pnl;
      }
    }
    
    // Calculate P&L percentage
    const dailyPnLPercentage = (dailyPnL / accountBalance) * 100;

    return {
      dailyPnL,
      dailyPnLPercentage
    };
  }

  /**
   * Generate risk alerts
   * @private
   * @param userId - User ID
   * @param metrics - Risk metrics
   * @returns Risk alerts
   */
  private async _generateAlerts(userId: string, metrics: any): Promise<RiskAlert[]> {
    const alerts: RiskAlert[] = [];
    
    // Check margin level
    if (metrics.marginLevel < 150) {
      alerts.push(await this.createAlert(
        userId,
        RiskAlertType.MARGIN_CALL,
        metrics.marginLevel < 120 ? 'critical' : 'warning',
        `Margin level is getting low: ${metrics.marginLevel.toFixed(2)}%`,
        { marginLevel: metrics.marginLevel }
      ));
    }
    
    // Check drawdown
    if (metrics.currentDrawdown > 10) {
      alerts.push(await this.createAlert(
        userId,
        RiskAlertType.DRAWDOWN_WARNING,
        metrics.currentDrawdown > 15 ? 'critical' : 'warning',
        `Account drawdown is significant: ${metrics.currentDrawdown.toFixed(2)}%`,
        { drawdown: metrics.currentDrawdown }
      ));
    }
    
    // Check daily loss
    if (metrics.dailyPnLPercentage < -3) {
      alerts.push(await this.createAlert(
        userId,
        RiskAlertType.DAILY_LOSS_WARNING,
        metrics.dailyPnLPercentage < -5 ? 'critical' : 'warning',
        `Significant daily loss: ${Math.abs(metrics.dailyPnLPercentage).toFixed(2)}%`,
        { dailyLoss: metrics.dailyPnLPercentage }
      ));
    }
    
    // Check high exposure
    for (const [symbol, exposure] of Object.entries(metrics.exposureBySymbol)) {
      if ((exposure as number) > 20) {
        alerts.push(await this.createAlert(
          userId,
          RiskAlertType.HIGH_EXPOSURE,
          (exposure as number) > 30 ? 'critical' : 'warning',
          `High exposure to ${symbol}: ${(exposure as number).toFixed(2)}%`,
          { symbol, exposure }
        ));
      }
    }
    
    // Check concentration risk
    const symbolCount = Object.keys(metrics.exposureBySymbol).length;
    if (symbolCount === 1 && metrics.totalRisk > 10) {
      alerts.push(await this.createAlert(
        userId,
        RiskAlertType.CONCENTRATION_RISK,
        'warning',
        'Portfolio is concentrated in a single asset',
        { symbolCount, totalRisk: metrics.totalRisk }
      ));
    }
    
    return alerts;
  }

  /**
   * Determine overall risk level
   * @private
   * @param metrics - Risk metrics
   * @returns Risk level
   */
  private _determineRiskLevel(metrics: any): RiskLevel {
    // Calculate a risk score based on various factors
    let riskScore = 0;
    
    // Margin level contribution
    if (metrics.marginLevel < 110) {
      riskScore += 40;
    } else if (metrics.marginLevel < 150) {
      riskScore += 30;
    } else if (metrics.marginLevel < 200) {
      riskScore += 20;
    } else if (metrics.marginLevel < 300) {
      riskScore += 10;
    }
    
    // Drawdown contribution
    if (metrics.currentDrawdown > 20) {
      riskScore += 40;
    } else if (metrics.currentDrawdown > 15) {
      riskScore += 30;
    } else if (metrics.currentDrawdown > 10) {
      riskScore += 20;
    } else if (metrics.currentDrawdown > 5) {
      riskScore += 10;
    }
    
    // Daily P&L contribution
    if (metrics.dailyPnLPercentage < -7) {
      riskScore += 30;
    } else if (metrics.dailyPnLPercentage < -5) {
      riskScore += 20;
    } else if (metrics.dailyPnLPercentage < -3) {
      riskScore += 10;
    }
    
    // Total risk contribution
    if (metrics.totalRisk > 30) {
      riskScore += 40;
    } else if (metrics.totalRisk > 20) {
      riskScore += 30;
    } else if (metrics.totalRisk > 15) {
      riskScore += 20;
    } else if (metrics.totalRisk > 10) {
      riskScore += 10;
    }
    
    // Determine risk level based on score
    if (riskScore >= 80) {
      return RiskLevel.VERY_HIGH;
    } else if (riskScore >= 60) {
      return RiskLevel.HIGH;
    } else if (riskScore >= 40) {
      return RiskLevel.MODERATE;
    } else if (riskScore >= 20) {
      return RiskLevel.LOW;
    } else {
      return RiskLevel.VERY_LOW;
    }
  }

  /**
   * Determine position risk level
   * @private
   * @param riskScore - Risk score
   * @returns Risk level
   */
  private _determinePositionRiskLevel(riskScore: number): RiskLevel {
    if (riskScore >= 80) {
      return RiskLevel.VERY_HIGH;
    } else if (riskScore >= 60) {
      return RiskLevel.HIGH;
    } else if (riskScore >= 40) {
      return RiskLevel.MODERATE;
    } else if (riskScore >= 20) {
      return RiskLevel.LOW;
    } else {
      return RiskLevel.VERY_LOW;
    }
  }
}

// Create default instance
const riskAssessmentService = new RiskAssessmentService();

export default riskAssessmentService; 