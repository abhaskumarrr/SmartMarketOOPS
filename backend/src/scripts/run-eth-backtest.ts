#!/usr/bin/env node

/**
 * ETH Intelligent Trading Bot Backtest
 * Execute backtest with $10 capital, 200x leverage on ETH using 1 year data
 */

import { IntelligentTradingBotBacktester } from '../backtesting/IntelligentTradingBotBacktester';
import { logger } from '../utils/logger';
import fs from 'fs';
import path from 'path';

async function runETHBacktest() {
  try {
    console.log('🚀 Starting ETH Intelligent Trading Bot Backtest');
    console.log('⚡ EXTREME PARAMETERS: $10 capital with 200x leverage!');
    logger.info('🚀 Starting ETH Intelligent Trading Bot Backtest');
    logger.info('⚡ EXTREME PARAMETERS: $10 capital with 200x leverage!');
    
    // Configure backtest parameters
    const config = {
      symbol: 'ETHUSD',
      startDate: '2023-01-01', // 1 year ago
      endDate: '2023-12-31',   // End of 2023
      initialCapital: 10,      // $10 starting capital
      leverage: 200,           // 200x leverage (EXTREME!)
      riskPerTrade: 40,        // 40% risk per trade (ULTRA EXTREME!)
      maxPositions: 1,         // Only 1 position at a time due to extreme leverage
      timeframe: '1h'          // 1-hour timeframe
    };

    logger.info('\n📋 ADVANCED SIGNAL FILTERING + DYNAMIC RISK LADDER:');
    logger.info(`💰 Initial Capital: $${config.initialCapital}`);
    logger.info(`⚡ Dynamic Leverage: 200x → 100x → 50x → 20x → 10x`);
    logger.info(`🎯 Dynamic Risk: 40% → 25% → 15% → 8% → 3%`);
    logger.info(`📊 Symbol: ${config.symbol}`);
    logger.info(`📅 Period: ${config.startDate} to ${config.endDate}`);
    logger.info(`🔄 Max Positions: ${config.maxPositions}`);

    logger.info('\n🔍 ADVANCED SIGNAL FILTERING (TARGET: 85%+ WIN RATE):');
    logger.info('🤖 ML Confidence Filter: 85%+ required');
    logger.info('📊 Ensemble Confidence: 80%+ required');
    logger.info('⏰ Timeframe Alignment: 75%+ (4+ timeframes)');
    logger.info('🌊 Regime Filter: Only trending/breakout markets');
    logger.info('📈 Technical Filter: RSI 25-75, Volume 1.3x, Low volatility');
    logger.info('🛡️ Risk Filter: Max 30% correlation, 15% drawdown limit');
    logger.info('🎯 Signal Score: 85+/100 required to trade');

    logger.info('\n🎯 RISK LADDER PHASES:');
    logger.info('📈 Phase 1: SURVIVAL ($10-$50) - 40% risk, 200x leverage');
    logger.info('🚀 Phase 2: GROWTH ($50-$200) - 25% risk, 100x leverage');
    logger.info('💪 Phase 3: EXPANSION ($200-$1K) - 15% risk, 50x leverage');
    logger.info('🛡️ Phase 4: CONSOLIDATION ($1K-$5K) - 8% risk, 20x leverage');
    logger.info('💎 Phase 5: PRESERVATION ($5K+) - 3% risk, 10x leverage');

    // Initialize backtester
    const backtester = new IntelligentTradingBotBacktester(config);
    
    // Run backtest
    const startTime = Date.now();
    const results = await backtester.runBacktest();
    const executionTime = Date.now() - startTime;
    
    // Display detailed results
    displayDetailedResults(results, executionTime);
    
    // Save results to file
    await saveBacktestResults(results);
    
    // Generate performance analysis
    generatePerformanceAnalysis(results);
    
    logger.info('✅ Backtest completed successfully!');
    
  } catch (error) {
    logger.error('❌ Backtest failed:', error);
    throw error;
  }
}

function displayDetailedResults(results: any, executionTime: number): void {
  logger.info('\n🎯 DETAILED BACKTEST RESULTS:');
  logger.info('═══════════════════════════════════════════════════════════');
  
  // Performance Summary
  logger.info('\n📊 PERFORMANCE SUMMARY:');
  logger.info(`💰 Starting Capital: $${results.config.initialCapital.toFixed(2)}`);
  logger.info(`💰 Final Balance: $${results.summary.finalBalance.toFixed(2)}`);
  logger.info(`📈 Total Return: $${results.summary.totalReturn.toFixed(2)}`);
  logger.info(`📊 Return Percentage: ${results.summary.totalReturnPercent.toFixed(2)}%`);
  logger.info(`⚡ With ${results.config.leverage}x leverage!`);
  
  // Risk Metrics
  logger.info('\n⚠️ RISK METRICS:');
  logger.info(`📉 Maximum Drawdown: ${results.summary.maxDrawdownPercent.toFixed(2)}%`);
  logger.info(`⚡ Sharpe Ratio: ${results.summary.sharpeRatio.toFixed(3)}`);
  logger.info(`🎯 Profit Factor: ${results.summary.profitFactor.toFixed(2)}`);
  
  // Trading Statistics
  logger.info('\n📈 TRADING STATISTICS:');
  logger.info(`🔢 Total Trades: ${results.summary.totalTrades}`);
  logger.info(`✅ Winning Trades: ${results.summary.winningTrades}`);
  logger.info(`❌ Losing Trades: ${results.summary.losingTrades}`);
  logger.info(`🎯 Win Rate: ${results.summary.winRate.toFixed(1)}%`);
  logger.info(`⏱️ Average Hold Time: ${results.summary.averageHoldTime.toFixed(1)} hours`);
  
  // P&L Analysis
  logger.info('\n💰 P&L ANALYSIS:');
  logger.info(`🏆 Average Win: $${results.summary.averageWin.toFixed(2)}`);
  logger.info(`💥 Average Loss: $${results.summary.averageLoss.toFixed(2)}`);
  logger.info(`🚀 Largest Win: $${results.summary.largestWin.toFixed(2)}`);
  logger.info(`💀 Largest Loss: $${results.summary.largestLoss.toFixed(2)}`);
  
  // Execution Info
  logger.info('\n⚡ EXECUTION INFO:');
  logger.info(`🕐 Execution Time: ${(executionTime / 1000).toFixed(2)} seconds`);
  logger.info(`📊 Data Points Processed: 365 days`);
  
  // Performance Rating
  const performanceRating = getPerformanceRating(results.summary.totalReturnPercent, results.summary.maxDrawdownPercent);
  logger.info(`\n🏆 PERFORMANCE RATING: ${performanceRating.rating}`);
  logger.info(`💡 ${performanceRating.comment}`);
  
  // Risk Warning
  if (results.config.leverage >= 100) {
    logger.info('\n⚠️ EXTREME LEVERAGE WARNING:');
    logger.info('🚨 This backtest uses EXTREME leverage (200x)!');
    logger.info('💀 Real trading with such leverage is EXTREMELY RISKY!');
    logger.info('📚 This is for educational/research purposes only!');
  }
}

function getPerformanceRating(returnPercent: number, maxDrawdown: number): { rating: string; comment: string } {
  if (returnPercent > 500 && maxDrawdown < 50) {
    return { rating: '🌟 EXCEPTIONAL', comment: 'Outstanding performance with controlled risk!' };
  } else if (returnPercent > 200 && maxDrawdown < 70) {
    return { rating: '🔥 EXCELLENT', comment: 'Strong returns with acceptable risk levels.' };
  } else if (returnPercent > 50 && maxDrawdown < 80) {
    return { rating: '✅ GOOD', comment: 'Solid performance, room for improvement.' };
  } else if (returnPercent > 0) {
    return { rating: '⚠️ MODERATE', comment: 'Profitable but needs optimization.' };
  } else {
    return { rating: '❌ POOR', comment: 'Strategy needs significant improvements.' };
  }
}

async function saveBacktestResults(results: any): Promise<void> {
  try {
    const resultsDir = path.join(__dirname, '../../results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `eth-backtest-${timestamp}.json`;
    const filepath = path.join(resultsDir, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(results, null, 2));
    logger.info(`💾 Results saved to: ${filepath}`);
    
    // Also save a summary CSV
    const csvData = generateCSVSummary(results);
    const csvFilename = `eth-backtest-summary-${timestamp}.csv`;
    const csvFilepath = path.join(resultsDir, csvFilename);
    fs.writeFileSync(csvFilepath, csvData);
    logger.info(`📊 CSV summary saved to: ${csvFilepath}`);
    
  } catch (error) {
    logger.error('❌ Error saving results:', error);
  }
}

function generateCSVSummary(results: any): string {
  const summary = results.summary;
  const config = results.config;
  
  let csv = 'Metric,Value\n';
  csv += `Symbol,${config.symbol}\n`;
  csv += `Initial Capital,$${config.initialCapital}\n`;
  csv += `Leverage,${config.leverage}x\n`;
  csv += `Final Balance,$${summary.finalBalance.toFixed(2)}\n`;
  csv += `Total Return,$${summary.totalReturn.toFixed(2)}\n`;
  csv += `Return Percentage,${summary.totalReturnPercent.toFixed(2)}%\n`;
  csv += `Total Trades,${summary.totalTrades}\n`;
  csv += `Win Rate,${summary.winRate.toFixed(1)}%\n`;
  csv += `Max Drawdown,${summary.maxDrawdownPercent.toFixed(2)}%\n`;
  csv += `Sharpe Ratio,${summary.sharpeRatio.toFixed(3)}\n`;
  csv += `Profit Factor,${summary.profitFactor.toFixed(2)}\n`;
  csv += `Average Hold Time,${summary.averageHoldTime.toFixed(1)} hours\n`;
  csv += `Largest Win,$${summary.largestWin.toFixed(2)}\n`;
  csv += `Largest Loss,$${summary.largestLoss.toFixed(2)}\n`;
  
  return csv;
}

function generatePerformanceAnalysis(results: any): void {
  logger.info('\n📊 PERFORMANCE ANALYSIS:');
  logger.info('═══════════════════════════════════════════════');
  
  // Monthly Performance
  if (results.monthlyBreakdown && results.monthlyBreakdown.length > 0) {
    logger.info('\n📅 MONTHLY BREAKDOWN:');
    results.monthlyBreakdown.forEach((month: any) => {
      const profitLoss = month.pnl >= 0 ? '📈' : '📉';
      logger.info(`${profitLoss} ${month.month}: ${month.trades} trades, $${month.pnl.toFixed(2)} P&L, ${month.winRate.toFixed(1)}% win rate`);
    });
  }
  
  // Regime Performance
  if (results.regimePerformance && results.regimePerformance.length > 0) {
    logger.info('\n🌊 REGIME PERFORMANCE:');
    results.regimePerformance.forEach((regime: any) => {
      logger.info(`📊 ${regime.regime}: ${regime.trades} trades, ${regime.winRate.toFixed(1)}% win rate, ${regime.avgReturn.toFixed(2)}% avg return`);
    });
  }
  
  // Risk Analysis
  logger.info('\n⚠️ RISK ANALYSIS:');
  const riskScore = calculateRiskScore(results.summary);
  logger.info(`🎯 Risk Score: ${riskScore.score}/100 (${riskScore.level})`);
  logger.info(`💡 Risk Assessment: ${riskScore.assessment}`);
  
  // Recommendations
  logger.info('\n💡 RECOMMENDATIONS:');
  const recommendations = generateRecommendations(results.summary);
  recommendations.forEach((rec: string, index: number) => {
    logger.info(`${index + 1}. ${rec}`);
  });
}

function calculateRiskScore(summary: any): { score: number; level: string; assessment: string } {
  let score = 100;
  
  // Penalize high drawdown
  if (summary.maxDrawdownPercent > 80) score -= 40;
  else if (summary.maxDrawdownPercent > 50) score -= 20;
  else if (summary.maxDrawdownPercent > 30) score -= 10;
  
  // Penalize low win rate
  if (summary.winRate < 30) score -= 30;
  else if (summary.winRate < 50) score -= 15;
  
  // Penalize low profit factor
  if (summary.profitFactor < 1) score -= 25;
  else if (summary.profitFactor < 1.5) score -= 10;
  
  score = Math.max(0, score);
  
  let level = 'LOW';
  let assessment = 'High risk strategy';
  
  if (score >= 80) {
    level = 'LOW';
    assessment = 'Well-controlled risk with good performance';
  } else if (score >= 60) {
    level = 'MEDIUM';
    assessment = 'Moderate risk, some areas for improvement';
  } else if (score >= 40) {
    level = 'HIGH';
    assessment = 'High risk, significant improvements needed';
  } else {
    level = 'EXTREME';
    assessment = 'Extremely high risk, major strategy overhaul required';
  }
  
  return { score, level, assessment };
}

function generateRecommendations(summary: any): string[] {
  const recommendations: string[] = [];
  
  if (summary.maxDrawdownPercent > 50) {
    recommendations.push('🛡️ Reduce position sizes or leverage to control drawdown');
  }
  
  if (summary.winRate < 50) {
    recommendations.push('🎯 Improve entry signals to increase win rate');
  }
  
  if (summary.profitFactor < 1.5) {
    recommendations.push('💰 Optimize take profit and stop loss levels');
  }
  
  if (summary.averageHoldTime > 24) {
    recommendations.push('⏱️ Consider shorter holding periods for high leverage trading');
  }
  
  if (summary.totalTrades < 50) {
    recommendations.push('📊 Increase trading frequency for better statistical significance');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('🎉 Excellent performance! Consider live testing with small amounts');
  }
  
  return recommendations;
}

// Execute backtest
if (require.main === module) {
  runETHBacktest().catch(error => {
    logger.error('❌ Backtest execution failed:', error);
    process.exit(1);
  });
}

export { runETHBacktest };
