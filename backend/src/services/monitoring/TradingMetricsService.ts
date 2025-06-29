/**
 * Phase 4: Production Readiness - Trading Metrics Service
 * Real-time tracking of trading performance and risk metrics
 */

import { EventEmitter } from 'events';
import prisma from '../../config/prisma-readonly';
import Redis from 'ioredis';
import { logger } from '../../utils/logger';

export interface TradingMetrics {
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalPnl: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  recoveryFactor: number;
  calmarRatio: number;
  sortinoRatio: number;
  valueAtRisk: number;
  expectedShortfall: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  treynorRatio: number;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
  databaseConnections: number;
  activeWebsockets: number;
  orderProcessingTime: number;
  apiResponseTime: number;
  errorRate: number;
  throughput: number;
}

export interface AlertThresholds {
  maxDrawdown: number;
  maxConsecutiveLosses: number;
  minWinRate: number;
  maxDailyLoss: number;
  maxErrorRate: number;
  maxResponseTime: number;
  minSharpeRatio: number;
  maxVaR: number;
}

export class TradingMetricsService extends EventEmitter {
  private redis: Redis;
  private metricsCache: Map<string, any> = new Map();
  private alertThresholds: AlertThresholds;
  private intervalId: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      maxRetriesPerRequest: null,
    });

    this.alertThresholds = {
      maxDrawdown: 0.2, // 20%
      maxConsecutiveLosses: 5,
      minWinRate: 0.4, // 40%
      maxDailyLoss: 0.05, // 5%
      maxErrorRate: 0.01, // 1%
      maxResponseTime: 1000, // 1 second
      minSharpeRatio: 0.5,
      maxVaR: 0.02, // 2%
    };

    this.startMetricsCollection();
  }

  /**
   * Start real-time metrics collection
   */
  private startMetricsCollection(): void {
    // Collect metrics every 30 seconds
    this.intervalId = setInterval(async () => {
      try {
        await this.collectAllMetrics();
      } catch (error) {
        logger.error('Error collecting metrics:', error);
      }
    }, 30000);

    logger.info('Trading metrics collection started');
  }

  /**
   * Stop metrics collection
   */
  public stopMetricsCollection(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    logger.info('Trading metrics collection stopped');
  }

  /**
   * Calculate comprehensive trading metrics for a user
   */
  public async calculateTradingMetrics(
    userId: string, 
    period: string = '30d'
  ): Promise<TradingMetrics> {
    try {
      const cacheKey = `metrics:${userId}:${period}`;
      const cached = await this.redis.get(cacheKey);
      
      if (cached) {
        return JSON.parse(cached);
      }

      const startDate = this.getPeriodStartDate(period);
      
      // Get all completed trades for the period
      const trades = await prisma.order.findMany({
        where: {
          userId,
          status: 'filled',
          executedAt: {
            gte: startDate,
          },
        },
        orderBy: { executedAt: 'asc' },
        select: {
          id: true,
          pnl: true,
          executedAt: true,
          symbol: true,
          side: true,
          quantity: true,
          executedPrice: true,
        },
      });

      if (trades.length === 0) {
        return this.getDefaultMetrics();
      }

      const metrics = this.computeMetrics(trades);
      
      // Cache for 5 minutes
      await this.redis.setex(cacheKey, 300, JSON.stringify(metrics));
      
      // Check for alerts
      await this.checkAlerts(userId, metrics);
      
      return metrics;
    } catch (error) {
      logger.error('Error calculating trading metrics:', error);
      throw error;
    }
  }

  /**
   * Calculate system performance metrics
   */
  public async calculateSystemMetrics(): Promise<SystemMetrics> {
    try {
      const cacheKey = 'system:metrics';
      const cached = await this.redis.get(cacheKey);
      
      if (cached) {
        return JSON.parse(cached);
      }

      const metrics: SystemMetrics = {
        cpuUsage: await this.getCpuUsage(),
        memoryUsage: await this.getMemoryUsage(),
        diskUsage: await this.getDiskUsage(),
        networkLatency: await this.getNetworkLatency(),
        databaseConnections: await this.getDatabaseConnections(),
        activeWebsockets: await this.getActiveWebsockets(),
        orderProcessingTime: await this.getOrderProcessingTime(),
        apiResponseTime: await this.getApiResponseTime(),
        errorRate: await this.getErrorRate(),
        throughput: await this.getThroughput(),
      };

      // Cache for 30 seconds
      await this.redis.setex(cacheKey, 30, JSON.stringify(metrics));
      
      return metrics;
    } catch (error) {
      logger.error('Error calculating system metrics:', error);
      throw error;
    }
  }

  /**
   * Get real-time trading performance dashboard data
   */
  public async getDashboardMetrics(userId: string): Promise<{
    trading: TradingMetrics;
    system: SystemMetrics;
    realTime: any;
  }> {
    const [trading, system, realTime] = await Promise.all([
      this.calculateTradingMetrics(userId),
      this.calculateSystemMetrics(),
      this.getRealTimeMetrics(userId),
    ]);

    return { trading, system, realTime };
  }

  /**
   * Track order execution metrics
   */
  public async trackOrderExecution(orderId: string, executionTime: number): Promise<void> {
    try {
      const key = 'metrics:order_execution_times';
      await this.redis.lpush(key, executionTime.toString());
      await this.redis.ltrim(key, 0, 999); // Keep last 1000 executions
      
      // Update average execution time
      const times = await this.redis.lrange(key, 0, -1);
      const averageTime = times.reduce((sum, time) => sum + parseFloat(time), 0) / times.length;
      
      await this.redis.set('metrics:avg_order_execution_time', averageTime.toString());

      logger.debug(`Order execution tracked: ${orderId}, time: ${executionTime}ms`);
    } catch (error) {
      logger.error('Error tracking order execution:', error);
    }
  }

  /**
   * Track API response times
   */
  public async trackApiResponse(endpoint: string, responseTime: number): Promise<void> {
    try {
      const key = `metrics:api_response_times:${endpoint}`;
      await this.redis.lpush(key, responseTime.toString());
      await this.redis.ltrim(key, 0, 999);

      // Update endpoint-specific average
      const times = await this.redis.lrange(key, 0, -1);
      const averageTime = times.reduce((sum, time) => sum + parseFloat(time), 0) / times.length;
      
      await this.redis.set(`metrics:avg_api_response_time:${endpoint}`, averageTime.toString());
    } catch (error) {
      logger.error('Error tracking API response time:', error);
    }
  }

  /**
   * Record trading signal quality metrics
   */
  public async trackSignalQuality(signalId: string, actualOutcome: 'win' | 'loss', confidence: number): Promise<void> {
    try {
      const key = 'metrics:signal_quality';
      const data = {
        signalId,
        outcome: actualOutcome,
        confidence,
        timestamp: Date.now(),
      };

      await this.redis.lpush(key, JSON.stringify(data));
      await this.redis.ltrim(key, 0, 9999); // Keep last 10k signals

      // Calculate signal accuracy
      const signals = await this.redis.lrange(key, 0, -1);
      const parsed = signals.map(s => JSON.parse(s));
      const accuracy = parsed.filter(s => s.outcome === 'win').length / parsed.length;
      
      await this.redis.set('metrics:signal_accuracy', accuracy.toString());

      logger.debug(`Signal quality tracked: ${signalId}, outcome: ${actualOutcome}, confidence: ${confidence}`);
    } catch (error) {
      logger.error('Error tracking signal quality:', error);
    }
  }

  /**
   * Generate performance report
   */
  public async generatePerformanceReport(userId: string, period: string): Promise<any> {
    try {
      const metrics = await this.calculateTradingMetrics(userId, period);
      const trades = await this.getTradesForPeriod(userId, period);
      
      const report = {
        generatedAt: new Date().toISOString(),
        period,
        userId,
        summary: {
          totalTrades: metrics.totalTrades,
          winRate: metrics.winRate,
          totalPnl: metrics.totalPnl,
          sharpeRatio: metrics.sharpeRatio,
          maxDrawdown: metrics.maxDrawdown,
        },
        performance: metrics,
        trades: trades.map(trade => ({
          id: trade.id,
          symbol: trade.symbol,
          side: trade.side,
          pnl: trade.pnl,
          executedAt: trade.executedAt,
        })),
        charts: {
          equityCurve: await this.generateEquityCurve(trades),
          drawdownChart: await this.generateDrawdownChart(trades),
          monthlyReturns: await this.generateMonthlyReturns(trades),
        },
        recommendations: await this.generateRecommendations(metrics),
      };

      return report;
    } catch (error) {
      logger.error('Error generating performance report:', error);
      throw error;
    }
  }

  /**
   * Private helper methods
   */
  private computeMetrics(trades: any[]): TradingMetrics {
    const wins = trades.filter(t => t.pnl > 0);
    const losses = trades.filter(t => t.pnl < 0);
    
    const totalPnl = trades.reduce((sum, t) => sum + t.pnl, 0);
    const winRate = wins.length / trades.length;
    const averageWin = wins.length > 0 ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0;
    const averageLoss = losses.length > 0 ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0;
    const profitFactor = averageLoss !== 0 ? Math.abs(averageWin / averageLoss) : 0;
    
    // Calculate returns for Sharpe ratio
    const returns = this.calculateReturns(trades);
    const sharpeRatio = this.calculateSharpeRatio(returns);
    const maxDrawdown = this.calculateMaxDrawdown(trades);
    
    // Consecutive wins/losses
    let consecutiveWins = 0;
    let consecutiveLosses = 0;
    let currentStreak = 0;
    let currentStreakType: 'win' | 'loss' | null = null;
    
    for (const trade of trades) {
      if (trade.pnl > 0) {
        if (currentStreakType === 'win') {
          currentStreak++;
        } else {
          currentStreak = 1;
          currentStreakType = 'win';
        }
        consecutiveWins = Math.max(consecutiveWins, currentStreak);
      } else if (trade.pnl < 0) {
        if (currentStreakType === 'loss') {
          currentStreak++;
        } else {
          currentStreak = 1;
          currentStreakType = 'loss';
        }
        consecutiveLosses = Math.max(consecutiveLosses, currentStreak);
      }
    }

    return {
      winRate,
      sharpeRatio,
      maxDrawdown,
      totalPnl,
      profitFactor,
      averageWin,
      averageLoss,
      totalTrades: trades.length,
      winningTrades: wins.length,
      losingTrades: losses.length,
      largestWin: wins.length > 0 ? Math.max(...wins.map(t => t.pnl)) : 0,
      largestLoss: losses.length > 0 ? Math.min(...losses.map(t => t.pnl)) : 0,
      consecutiveWins,
      consecutiveLosses,
      recoveryFactor: maxDrawdown !== 0 ? totalPnl / Math.abs(maxDrawdown) : 0,
      calmarRatio: maxDrawdown !== 0 ? totalPnl / Math.abs(maxDrawdown) : 0,
      sortinoRatio: this.calculateSortinoRatio(returns),
      valueAtRisk: this.calculateVaR(returns, 0.05),
      expectedShortfall: this.calculateExpectedShortfall(returns, 0.05),
      beta: 0, // Would need market data to calculate
      alpha: 0, // Would need market data to calculate
      informationRatio: 0, // Would need benchmark to calculate
      treynorRatio: 0, // Would need beta to calculate
    };
  }

  private calculateReturns(trades: any[]): number[] {
    if (trades.length === 0) return [];
    
    let cumulativeValue = 100000; // Starting balance
    const returns: number[] = [];
    
    for (const trade of trades) {
      const previousValue = cumulativeValue;
      cumulativeValue += trade.pnl;
      const dailyReturn = (cumulativeValue - previousValue) / previousValue;
      returns.push(dailyReturn);
    }
    
    return returns;
  }

  private calculateSharpeRatio(returns: number[], riskFreeRate: number = 0.02): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    return stdDev === 0 ? 0 : (avgReturn - riskFreeRate / 252) / stdDev;
  }

  private calculateMaxDrawdown(trades: any[]): number {
    if (trades.length === 0) return 0;
    
    let cumulativeValue = 100000;
    let peak = cumulativeValue;
    let maxDrawdown = 0;
    
    for (const trade of trades) {
      cumulativeValue += trade.pnl;
      
      if (cumulativeValue > peak) {
        peak = cumulativeValue;
      }
      
      const drawdown = (peak - cumulativeValue) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    
    return maxDrawdown;
  }

  private calculateSortinoRatio(returns: number[], riskFreeRate: number = 0.02): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const negativeReturns = returns.filter(r => r < 0);
    
    if (negativeReturns.length === 0) return Infinity;
    
    const downwardVariance = negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length;
    const downwardStdDev = Math.sqrt(downwardVariance);
    
    return downwardStdDev === 0 ? 0 : (avgReturn - riskFreeRate / 252) / downwardStdDev;
  }

  private calculateVaR(returns: number[], confidence: number): number {
    if (returns.length === 0) return 0;
    
    const sorted = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sorted.length);
    
    return sorted[index] || 0;
  }

  private calculateExpectedShortfall(returns: number[], confidence: number): number {
    if (returns.length === 0) return 0;
    
    const var_ = this.calculateVaR(returns, confidence);
    const tail = returns.filter(r => r <= var_);
    
    return tail.length > 0 ? tail.reduce((sum, r) => sum + r, 0) / tail.length : 0;
  }

  private getPeriodStartDate(period: string): Date {
    const now = new Date();
    
    switch (period) {
      case '1d':
        return new Date(now.getTime() - 24 * 60 * 60 * 1000);
      case '7d':
        return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      case '30d':
        return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      case '90d':
        return new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
      case '1y':
        return new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
      default:
        return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    }
  }

  private getDefaultMetrics(): TradingMetrics {
    return {
      winRate: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      totalPnl: 0,
      profitFactor: 0,
      averageWin: 0,
      averageLoss: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      largestWin: 0,
      largestLoss: 0,
      consecutiveWins: 0,
      consecutiveLosses: 0,
      recoveryFactor: 0,
      calmarRatio: 0,
      sortinoRatio: 0,
      valueAtRisk: 0,
      expectedShortfall: 0,
      beta: 0,
      alpha: 0,
      informationRatio: 0,
      treynorRatio: 0,
    };
  }

  private async checkAlerts(userId: string, metrics: TradingMetrics): Promise<void> {
    const alerts: string[] = [];

    if (metrics.maxDrawdown > this.alertThresholds.maxDrawdown) {
      alerts.push(`Max drawdown exceeded: ${(metrics.maxDrawdown * 100).toFixed(2)}%`);
    }

    if (metrics.consecutiveLosses > this.alertThresholds.maxConsecutiveLosses) {
      alerts.push(`Consecutive losses limit exceeded: ${metrics.consecutiveLosses}`);
    }

    if (metrics.winRate < this.alertThresholds.minWinRate && metrics.totalTrades > 10) {
      alerts.push(`Win rate below threshold: ${(metrics.winRate * 100).toFixed(2)}%`);
    }

    if (metrics.sharpeRatio < this.alertThresholds.minSharpeRatio && metrics.totalTrades > 20) {
      alerts.push(`Sharpe ratio below threshold: ${metrics.sharpeRatio.toFixed(2)}`);
    }

    if (alerts.length > 0) {
      this.emit('alert', { userId, alerts, metrics });
      logger.warn(`Trading alerts for user ${userId}:`, alerts);
    }
  }

  private async collectAllMetrics(): Promise<void> {
    // Collect system metrics
    const systemMetrics = await this.calculateSystemMetrics();
    await this.redis.setex('metrics:system:latest', 300, JSON.stringify(systemMetrics));

    // Emit metrics for real-time monitoring
    this.emit('metrics:system', systemMetrics);
  }

  // Placeholder methods for system metrics (would integrate with actual monitoring tools)
  private async getCpuUsage(): Promise<number> {
    // Integration with system monitoring
    return Math.random() * 100; // Placeholder
  }

  private async getMemoryUsage(): Promise<number> {
    const used = process.memoryUsage();
    return (used.heapUsed / 1024 / 1024); // MB
  }

  private async getDiskUsage(): Promise<number> {
    // Integration with system monitoring
    return Math.random() * 100; // Placeholder
  }

  private async getNetworkLatency(): Promise<number> {
    // Ping test or API latency check
    return Math.random() * 100; // Placeholder
  }

  private async getDatabaseConnections(): Promise<number> {
    // Check active database connections
    return Math.floor(Math.random() * 50);
  }

  private async getActiveWebsockets(): Promise<number> {
    const count = await this.redis.get('websocket:active_count') || '0';
    return parseInt(count);
  }

  private async getOrderProcessingTime(): Promise<number> {
    const avgTime = await this.redis.get('metrics:avg_order_execution_time');
    return avgTime ? parseFloat(avgTime) : 0;
  }

  private async getApiResponseTime(): Promise<number> {
    // Calculate average API response time across all endpoints
    const keys = await this.redis.keys('metrics:avg_api_response_time:*');
    if (keys.length === 0) return 0;

    const times = await Promise.all(keys.map(key => this.redis.get(key)));
    const validTimes = times.filter(time => time !== null).map(time => parseFloat(time!));
    
    return validTimes.length > 0 ? validTimes.reduce((sum, time) => sum + time, 0) / validTimes.length : 0;
  }

  private async getErrorRate(): Promise<number> {
    const errors = await this.redis.get('metrics:error_count') || '0';
    const requests = await this.redis.get('metrics:request_count') || '1';
    
    return parseInt(errors) / parseInt(requests);
  }

  private async getThroughput(): Promise<number> {
    const requests = await this.redis.get('metrics:request_count') || '0';
    return parseInt(requests) / 60; // requests per minute
  }

  private async getRealTimeMetrics(userId: string): Promise<any> {
    // Get real-time metrics for the user
    return {
      activePositions: await this.getActivePositionsCount(userId),
      pendingOrders: await this.getPendingOrdersCount(userId),
      todayPnl: await this.getTodayPnl(userId),
      currentDrawdown: await this.getCurrentDrawdown(userId),
    };
  }

  private async getActivePositionsCount(userId: string): Promise<number> {
    return await prisma.position.count({
      where: { userId, status: 'open' }
    });
  }

  private async getPendingOrdersCount(userId: string): Promise<number> {
    return await prisma.order.count({
      where: { userId, status: 'pending' }
    });
  }

  private async getTodayPnl(userId: string): Promise<number> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const result = await prisma.order.aggregate({
      where: {
        userId,
        status: 'filled',
        executedAt: { gte: today }
      },
      _sum: { pnl: true }
    });

    return result._sum.pnl || 0;
  }

  private async getCurrentDrawdown(userId: string): Promise<number> {
    // Calculate current drawdown from peak
    const trades = await prisma.order.findMany({
      where: { userId, status: 'filled' },
      orderBy: { executedAt: 'asc' },
      select: { pnl: true }
    });

    return this.calculateMaxDrawdown(trades);
  }

  private async getTradesForPeriod(userId: string, period: string): Promise<any[]> {
    const startDate = this.getPeriodStartDate(period);
    
    return await prisma.order.findMany({
      where: {
        userId,
        status: 'filled',
        executedAt: { gte: startDate }
      },
      orderBy: { executedAt: 'asc' }
    });
  }

  private async generateEquityCurve(trades: any[]): Promise<Array<{ date: string; value: number }>> {
    let cumulativeValue = 100000;
    const curve = [{ date: new Date().toISOString(), value: cumulativeValue }];

    for (const trade of trades) {
      cumulativeValue += trade.pnl;
      curve.push({
        date: trade.executedAt.toISOString(),
        value: cumulativeValue
      });
    }

    return curve;
  }

  private async generateDrawdownChart(trades: any[]): Promise<Array<{ date: string; drawdown: number }>> {
    let cumulativeValue = 100000;
    let peak = cumulativeValue;
    const chart = [];

    for (const trade of trades) {
      cumulativeValue += trade.pnl;
      
      if (cumulativeValue > peak) {
        peak = cumulativeValue;
      }
      
      const drawdown = (peak - cumulativeValue) / peak;
      chart.push({
        date: trade.executedAt.toISOString(),
        drawdown: -drawdown // Negative for chart display
      });
    }

    return chart;
  }

  private async generateMonthlyReturns(trades: any[]): Promise<Array<{ month: string; return: number }>> {
    const monthlyData = new Map<string, number>();
    let previousValue = 100000;

    for (const trade of trades) {
      const month = trade.executedAt.toISOString().substring(0, 7); // YYYY-MM
      const currentValue = previousValue + trade.pnl;
      const monthlyReturn = (currentValue - previousValue) / previousValue;
      
      monthlyData.set(month, (monthlyData.get(month) || 0) + monthlyReturn);
      previousValue = currentValue;
    }

    return Array.from(monthlyData.entries()).map(([month, return_]) => ({
      month,
      return: return_
    }));
  }

  private async generateRecommendations(metrics: TradingMetrics): Promise<string[]> {
    const recommendations: string[] = [];

    if (metrics.winRate < 0.5 && metrics.totalTrades > 10) {
      recommendations.push('Consider reviewing your entry criteria to improve win rate');
    }

    if (metrics.sharpeRatio < 1.0 && metrics.totalTrades > 20) {
      recommendations.push('Focus on risk-adjusted returns by improving your risk management');
    }

    if (metrics.maxDrawdown > 0.15) {
      recommendations.push('Reduce position sizes to limit maximum drawdown');
    }

    if (metrics.consecutiveLosses > 3) {
      recommendations.push('Consider implementing a pause mechanism after consecutive losses');
    }

    if (metrics.profitFactor < 1.5) {
      recommendations.push('Work on cutting losses faster and letting winners run longer');
    }

    return recommendations;
  }
}

export default TradingMetricsService;