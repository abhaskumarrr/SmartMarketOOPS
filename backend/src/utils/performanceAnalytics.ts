/**
 * Performance Analytics for Backtesting
 * Calculates comprehensive trading performance metrics
 */

import { 
  Trade, 
  PortfolioSnapshot, 
  PerformanceMetrics, 
  BacktestConfig 
} from '../types/marketData';
import { logger } from '../utils/logger';

export class PerformanceAnalytics {
  /**
   * Calculate comprehensive performance metrics
   */
  public static calculateMetrics(
    trades: Trade[], 
    portfolioHistory: PortfolioSnapshot[], 
    config: BacktestConfig
  ): PerformanceMetrics {
    logger.info('📊 Calculating performance metrics...');

    const finalPortfolio = portfolioHistory[portfolioHistory.length - 1];
    const initialCapital = config.initialCapital;
    
    // Basic returns
    const totalReturn = finalPortfolio.totalValue - initialCapital;
    const totalReturnPercent = (totalReturn / initialCapital) * 100;
    
    // Time-based calculations
    const startTime = portfolioHistory[0]?.timestamp || 0;
    const endTime = finalPortfolio.timestamp;
    const durationYears = (endTime - startTime) / (1000 * 60 * 60 * 24 * 365.25);
    const annualizedReturn = durationYears > 0 ? 
      (Math.pow(finalPortfolio.totalValue / initialCapital, 1 / durationYears) - 1) * 100 : 0;

    // Trade statistics
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    const totalTrades = trades.length;
    const winRate = totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0;

    // P&L statistics
    const totalWins = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const totalLosses = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
    const averageWin = winningTrades.length > 0 ? totalWins / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? totalLosses / losingTrades.length : 0;
    const averageWinPercent = winningTrades.length > 0 ? 
      winningTrades.reduce((sum, t) => sum + t.pnlPercent, 0) / winningTrades.length : 0;
    const averageLossPercent = losingTrades.length > 0 ? 
      Math.abs(losingTrades.reduce((sum, t) => sum + t.pnlPercent, 0) / losingTrades.length) : 0;

    // Risk metrics
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0;
    const payoffRatio = averageLoss > 0 ? averageWin / averageLoss : averageWin > 0 ? Infinity : 0;
    const expectancy = totalTrades > 0 ? totalReturn / totalTrades : 0;

    // Drawdown analysis
    const maxDrawdown = Math.max(...portfolioHistory.map(p => p.drawdown));
    const maxDrawdownPercent = maxDrawdown;

    // Find largest win and loss
    const largestWin = winningTrades.length > 0 ? Math.max(...winningTrades.map(t => t.pnl)) : 0;
    const largestLoss = losingTrades.length > 0 ? Math.min(...losingTrades.map(t => t.pnl)) : 0;

    // Average trade duration
    const averageTradeDuration = totalTrades > 0 ? 
      trades.reduce((sum, t) => sum + t.duration, 0) / totalTrades / (1000 * 60 * 60) : 0; // in hours

    // Volatility calculation
    const returns = this.calculateReturns(portfolioHistory);
    const volatility = this.calculateVolatility(returns) * 100; // Annualized volatility

    // Risk-adjusted returns
    const sharpeRatio = volatility > 0 ? annualizedReturn / volatility : 0;
    const sortinoRatio = this.calculateSortinoRatio(returns, 0) * 100;
    const calmarRatio = maxDrawdownPercent > 0 ? annualizedReturn / maxDrawdownPercent : 0;
    const recoveryFactor = maxDrawdownPercent > 0 ? totalReturnPercent / maxDrawdownPercent : 0;

    const metrics: PerformanceMetrics = {
      totalReturn,
      totalReturnPercent,
      annualizedReturn,
      sharpeRatio,
      sortinoRatio,
      maxDrawdown: maxDrawdown * initialCapital / 100, // Convert to dollar amount
      maxDrawdownPercent,
      winRate,
      profitFactor,
      averageWin,
      averageLoss,
      averageWinPercent,
      averageLossPercent,
      totalTrades,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      largestWin,
      largestLoss,
      averageTradeDuration,
      volatility,
      calmarRatio,
      recoveryFactor,
      payoffRatio,
      expectancy,
    };

    logger.info('✅ Performance metrics calculated', {
      totalReturn: totalReturnPercent.toFixed(2) + '%',
      winRate: winRate.toFixed(1) + '%',
      sharpeRatio: sharpeRatio.toFixed(2),
      maxDrawdown: maxDrawdownPercent.toFixed(2) + '%',
      totalTrades,
    });

    return metrics;
  }

  /**
   * Calculate period returns from portfolio history
   */
  private static calculateReturns(portfolioHistory: PortfolioSnapshot[]): number[] {
    const returns: number[] = [];
    
    for (let i = 1; i < portfolioHistory.length; i++) {
      const currentValue = portfolioHistory[i].totalValue;
      const previousValue = portfolioHistory[i - 1].totalValue;
      const periodReturn = (currentValue - previousValue) / previousValue;
      returns.push(periodReturn);
    }
    
    return returns;
  }

  /**
   * Calculate annualized volatility
   */
  private static calculateVolatility(returns: number[]): number {
    if (returns.length < 2) return 0;

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    const stdDev = Math.sqrt(variance);
    
    // Annualize assuming daily returns (multiply by sqrt of trading days per year)
    return stdDev * Math.sqrt(252);
  }

  /**
   * Calculate Sortino Ratio
   */
  private static calculateSortinoRatio(returns: number[], targetReturn: number = 0): number {
    if (returns.length < 2) return 0;

    const excessReturns = returns.map(r => r - targetReturn);
    const meanExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
    
    const downside = excessReturns.filter(r => r < 0);
    if (downside.length === 0) return Infinity;
    
    const downsideVariance = downside.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downside.length;
    const downsideDeviation = Math.sqrt(downsideVariance);
    
    if (downsideDeviation === 0) return Infinity;
    
    // Annualize
    return (meanExcessReturn * Math.sqrt(252)) / (downsideDeviation * Math.sqrt(252));
  }

  /**
   * Generate detailed performance report
   */
  public static generateReport(
    metrics: PerformanceMetrics, 
    trades: Trade[], 
    config: BacktestConfig
  ): string {
    const report = `
📊 BACKTESTING PERFORMANCE REPORT
${'='.repeat(50)}

🎯 STRATEGY: ${config.strategy}
📈 SYMBOL: ${config.symbol}
⏰ TIMEFRAME: ${config.timeframe}
📅 PERIOD: ${config.startDate.toISOString().split('T')[0]} to ${config.endDate.toISOString().split('T')[0]}

💰 CAPITAL & RISK MANAGEMENT
${'='.repeat(30)}
Initial Capital: $${config.initialCapital.toLocaleString()}
Leverage: ${config.leverage}x
Risk per Trade: ${config.riskPerTrade}%
Commission: ${config.commission}%

📈 OVERALL PERFORMANCE
${'='.repeat(30)}
Total Return: $${metrics.totalReturn.toLocaleString()} (${metrics.totalReturnPercent.toFixed(2)}%)
Annualized Return: ${metrics.annualizedReturn.toFixed(2)}%
Sharpe Ratio: ${metrics.sharpeRatio.toFixed(2)}
Sortino Ratio: ${metrics.sortinoRatio.toFixed(2)}
Calmar Ratio: ${metrics.calmarRatio.toFixed(2)}

📊 RISK METRICS
${'='.repeat(30)}
Maximum Drawdown: $${metrics.maxDrawdown.toLocaleString()} (${metrics.maxDrawdownPercent.toFixed(2)}%)
Volatility: ${metrics.volatility.toFixed(2)}%
Recovery Factor: ${metrics.recoveryFactor.toFixed(2)}

🎯 TRADING STATISTICS
${'='.repeat(30)}
Total Trades: ${metrics.totalTrades}
Winning Trades: ${metrics.winningTrades} (${metrics.winRate.toFixed(1)}%)
Losing Trades: ${metrics.losingTrades} (${(100 - metrics.winRate).toFixed(1)}%)
Profit Factor: ${metrics.profitFactor.toFixed(2)}
Payoff Ratio: ${metrics.payoffRatio.toFixed(2)}
Expectancy: $${metrics.expectancy.toFixed(2)}

💵 TRADE ANALYSIS
${'='.repeat(30)}
Average Win: $${metrics.averageWin.toFixed(2)} (${metrics.averageWinPercent.toFixed(2)}%)
Average Loss: $${metrics.averageLoss.toFixed(2)} (${metrics.averageLossPercent.toFixed(2)}%)
Largest Win: $${metrics.largestWin.toFixed(2)}
Largest Loss: $${metrics.largestLoss.toFixed(2)}
Average Trade Duration: ${metrics.averageTradeDuration.toFixed(1)} hours

⭐ PERFORMANCE RATING
${'='.repeat(30)}
${this.getPerformanceRating(metrics)}

📋 TRADE SUMMARY
${'='.repeat(30)}
${this.getTopTrades(trades, 5)}
`;

    return report;
  }

  /**
   * Get performance rating based on key metrics
   */
  private static getPerformanceRating(metrics: PerformanceMetrics): string {
    let score = 0;
    const ratings: string[] = [];

    // Return rating
    if (metrics.totalReturnPercent > 50) { score += 2; ratings.push('🟢 Excellent Returns'); }
    else if (metrics.totalReturnPercent > 20) { score += 1; ratings.push('🟡 Good Returns'); }
    else if (metrics.totalReturnPercent > 0) { score += 0; ratings.push('🟡 Positive Returns'); }
    else { score -= 1; ratings.push('🔴 Negative Returns'); }

    // Sharpe ratio rating
    if (metrics.sharpeRatio > 2) { score += 2; ratings.push('🟢 Excellent Risk-Adjusted Returns'); }
    else if (metrics.sharpeRatio > 1) { score += 1; ratings.push('🟡 Good Risk-Adjusted Returns'); }
    else if (metrics.sharpeRatio > 0) { score += 0; ratings.push('🟡 Positive Risk-Adjusted Returns'); }
    else { score -= 1; ratings.push('🔴 Poor Risk-Adjusted Returns'); }

    // Win rate rating
    if (metrics.winRate > 60) { score += 1; ratings.push('🟢 High Win Rate'); }
    else if (metrics.winRate > 40) { score += 0; ratings.push('🟡 Moderate Win Rate'); }
    else { score -= 1; ratings.push('🔴 Low Win Rate'); }

    // Drawdown rating
    if (metrics.maxDrawdownPercent < 10) { score += 1; ratings.push('🟢 Low Drawdown'); }
    else if (metrics.maxDrawdownPercent < 20) { score += 0; ratings.push('🟡 Moderate Drawdown'); }
    else { score -= 1; ratings.push('🔴 High Drawdown'); }

    let overall = '';
    if (score >= 4) overall = '🌟 EXCELLENT STRATEGY';
    else if (score >= 2) overall = '✅ GOOD STRATEGY';
    else if (score >= 0) overall = '⚠️ AVERAGE STRATEGY';
    else overall = '❌ POOR STRATEGY';

    return `${overall}\n\n${ratings.join('\n')}`;
  }

  /**
   * Get top performing trades
   */
  private static getTopTrades(trades: Trade[], count: number): string {
    const sortedTrades = [...trades].sort((a, b) => b.pnl - a.pnl);
    const topTrades = sortedTrades.slice(0, count);
    
    if (topTrades.length === 0) {
      return 'No trades executed.';
    }

    let summary = `Top ${Math.min(count, topTrades.length)} Trades:\n`;
    
    topTrades.forEach((trade, index) => {
      const duration = (trade.duration / (1000 * 60 * 60)).toFixed(1);
      summary += `${index + 1}. ${trade.side} ${trade.symbol}: $${trade.pnl.toFixed(2)} (${trade.pnlPercent.toFixed(2)}%) - ${duration}h\n`;
    });

    return summary;
  }
}
