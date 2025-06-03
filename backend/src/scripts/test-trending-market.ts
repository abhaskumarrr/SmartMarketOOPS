#!/usr/bin/env node

/**
 * Test Enhanced Strategy in Trending Market Conditions
 * Creates a trending market scenario to validate strategy performance
 */

import { marketDataService } from '../services/marketDataProvider';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { PortfolioManager } from '../services/portfolioManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { createEnhancedTrendStrategy } from '../strategies/enhancedTrendStrategy';
import { BacktestConfig, EnhancedMarketData, MarketDataPoint } from '../types/marketData';
import { logger } from '../utils/logger';

class TrendingMarketTest {
  /**
   * Test strategy in trending market conditions
   */
  public async runTest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('📈 Testing Enhanced Strategy in Trending Market...');

      // Create trending market data
      const trendingData = this.createTrendingMarketData();
      
      // Test with relaxed parameters for trending market
      const config = this.createTrendingConfig();
      const strategy = this.createOptimizedStrategy();
      
      // Run backtest
      const result = await this.testStrategy(config, strategy, trendingData);
      
      // Display results
      this.displayResults(result, startTime);

      logger.info('🎉 Trending market test completed!');

    } catch (error) {
      logger.error('❌ Trending market test failed:', error);
      throw error;
    }
  }

  /**
   * Create synthetic trending market data
   */
  private createTrendingMarketData(): EnhancedMarketData[] {
    logger.info('📊 Creating synthetic trending market data...');
    
    const dataPoints = 200; // 200 hours of data
    const startPrice = 45000;
    const trendStrength = 0.002; // 0.2% per hour uptrend
    const volatility = 0.015; // 1.5% volatility
    
    const data: MarketDataPoint[] = [];
    let currentPrice = startPrice;
    let currentTime = Date.now() - (dataPoints * 60 * 60 * 1000);
    
    for (let i = 0; i < dataPoints; i++) {
      // Strong uptrend with some volatility
      const trendMove = currentPrice * trendStrength;
      const randomMove = currentPrice * volatility * (Math.random() - 0.5) * 2;
      
      currentPrice += trendMove + randomMove;
      currentPrice = Math.max(currentPrice, 1000); // Floor price
      
      // Generate OHLC
      const open = currentPrice;
      const range = currentPrice * volatility * 0.3;
      const high = open + (Math.random() * range);
      const low = open - (Math.random() * range);
      const close = low + (Math.random() * (high - low));
      
      // Higher volume during trend
      const volume = 100 + (Math.random() * 150) + (i * 2); // Increasing volume
      
      data.push({
        timestamp: currentTime,
        symbol: 'BTCUSD',
        exchange: 'trending-mock',
        timeframe: '1h',
        open,
        high: Math.max(open, high, close),
        low: Math.min(open, low, close),
        close,
        volume,
      });
      
      currentTime += 60 * 60 * 1000; // 1 hour
      currentPrice = close;
    }

    // Enhance with indicators
    const closes = data.map(d => d.close);
    const volumes = data.map(d => d.volume);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);

    const sma20 = technicalAnalysis.calculateSMA(closes, 20);
    const sma50 = technicalAnalysis.calculateSMA(closes, 50);
    const ema12 = technicalAnalysis.calculateEMA(closes, 12);
    const ema26 = technicalAnalysis.calculateEMA(closes, 26);
    const rsi = technicalAnalysis.calculateRSI(closes, 14);
    const macd = technicalAnalysis.calculateMACD(closes, 12, 26, 9);
    const bollinger = technicalAnalysis.calculateBollingerBands(closes, 20, 2);
    const volumeSMA = technicalAnalysis.calculateSMA(volumes, 20);

    const enhancedData: EnhancedMarketData[] = data.map((point, index) => ({
      ...point,
      indicators: {
        sma_20: sma20[index],
        sma_50: sma50[index],
        ema_12: ema12[index],
        ema_26: ema26[index],
        rsi: rsi[index],
        macd: macd.macd[index],
        macd_signal: macd.signal[index],
        macd_histogram: macd.histogram[index],
        bollinger_upper: bollinger.upper[index],
        bollinger_middle: bollinger.middle[index],
        bollinger_lower: bollinger.lower[index],
        volume_sma: volumeSMA[index],
      },
    }));

    logger.info(`✅ Created ${dataPoints} trending market data points`);
    logger.info(`📈 Price movement: $${startPrice.toLocaleString()} → $${currentPrice.toLocaleString()} (+${((currentPrice/startPrice - 1) * 100).toFixed(1)}%)`);
    
    return enhancedData;
  }

  /**
   * Create config optimized for trending markets
   */
  private createTrendingConfig(): BacktestConfig {
    return {
      symbol: 'BTCUSD',
      timeframe: '1h',
      startDate: new Date(Date.now() - (200 * 60 * 60 * 1000)),
      endDate: new Date(),
      initialCapital: 2000,
      leverage: 3,
      riskPerTrade: 2,
      commission: 0.1,
      slippage: 0.05,
      strategy: 'Enhanced_Trend_Optimized',
      parameters: {},
    };
  }

  /**
   * Create strategy optimized for trending markets
   */
  private createOptimizedStrategy() {
    return createEnhancedTrendStrategy({
      // More aggressive for trending markets
      trendPeriod: 15,           // Shorter trend detection
      trendThreshold: 0.0005,    // Lower threshold for trend
      minTrendStrength: 0.4,     // Lower strength requirement
      minConfidence: 65,         // Lower confidence threshold
      antiWhipsawPeriod: 3,      // Shorter anti-whipsaw
      
      // Better risk management
      baseStopLoss: 1.2,         // Tighter stops
      takeProfitMultiplier: 4,   // Higher R:R ratio
      maxPositionSize: 0.9,      // Larger positions in trends
      
      // Volume confirmation
      volumeConfirmation: 1.3,   // Lower volume requirement
      
      // RSI settings for trending markets
      rsiOverbought: 80,         // Allow more overbought
      rsiOversold: 20,           // Allow more oversold
    });
  }

  /**
   * Test strategy with trending data
   */
  private async testStrategy(config: BacktestConfig, strategy: any, marketData: EnhancedMarketData[]): Promise<any> {
    strategy.initialize(config);
    const portfolioManager = new PortfolioManager(config);
    
    let signalCount = 0;
    let tradeCount = 0;
    let validSignals = 0;

    for (let i = 0; i < marketData.length; i++) {
      const currentCandle = marketData[i];

      // Update portfolio
      portfolioManager.updatePositions(
        currentCandle.symbol, 
        currentCandle.close, 
        currentCandle.timestamp
      );

      // Check stop loss/take profit
      const closedTrades = portfolioManager.checkStopLossAndTakeProfit(
        currentCandle.symbol,
        currentCandle.close,
        currentCandle.timestamp
      );

      tradeCount += closedTrades.length;

      // Generate signal
      const signal = strategy.generateSignal(marketData, i);
      
      if (signal) {
        signalCount++;
        
        if (signal.confidence > 0) {
          validSignals++;
          
          logger.info(`📊 Generated ${signal.type} signal at $${currentCandle.close.toFixed(0)} (Confidence: ${signal.confidence.toFixed(0)}%)`);
          
          // Execute trade
          const trade = portfolioManager.executeTrade(
            signal, 
            currentCandle.close, 
            currentCandle.timestamp
          );

          if (trade) {
            trade.strategy = 'Enhanced_Trend_Optimized';
            tradeCount++;
          }
        }
      }

      // Portfolio snapshots
      if (i % 24 === 0 || signal) {
        portfolioManager.createSnapshot(currentCandle.timestamp);
      }
    }

    // Calculate performance
    const trades = portfolioManager.getTrades();
    const portfolioHistory = portfolioManager.getPortfolioHistory();
    const performance = PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);

    return {
      config,
      performance,
      trades,
      portfolioHistory,
      dataPoints: marketData.length,
      signalCount,
      validSignals,
      tradeCount,
      strategy,
    };
  }

  /**
   * Display comprehensive results
   */
  private displayResults(result: any, startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;
    const performance = result.performance;

    logger.info('\n' + '📈 TRENDING MARKET TEST RESULTS'.padStart(50, '='));
    logger.info('=' .repeat(80));

    // Execution metrics
    logger.info('⚡ EXECUTION METRICS:');
    logger.info(`   Duration: ${duration.toFixed(2)} seconds`);
    logger.info(`   Data Points: ${result.dataPoints.toLocaleString()}`);
    logger.info(`   Processing Speed: ${(result.dataPoints / duration).toFixed(0)} points/sec`);
    logger.info(`   Signals Generated: ${result.signalCount}`);
    logger.info(`   Valid Signals: ${result.validSignals}`);
    logger.info(`   Trades Executed: ${result.tradeCount}`);

    // Performance metrics
    logger.info('\n💰 PERFORMANCE IN TRENDING MARKET:');
    logger.info(`   Total Return: $${performance.totalReturn.toFixed(2)} (${performance.totalReturnPercent.toFixed(2)}%)`);
    logger.info(`   Annualized Return: ${performance.annualizedReturn.toFixed(2)}%`);
    logger.info(`   Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
    logger.info(`   Maximum Drawdown: ${performance.maxDrawdownPercent.toFixed(2)}%`);
    logger.info(`   Win Rate: ${performance.winRate.toFixed(1)}%`);
    logger.info(`   Profit Factor: ${performance.profitFactor.toFixed(2)}`);
    logger.info(`   Average Win: $${performance.averageWin.toFixed(2)}`);
    logger.info(`   Average Loss: $${performance.averageLoss.toFixed(2)}`);

    // Strategy validation
    logger.info('\n🎯 STRATEGY VALIDATION:');
    if (performance.totalReturnPercent > 10) {
      logger.info('   ✅ Strong positive returns in trending market');
    }
    if (performance.winRate > 50) {
      logger.info('   ✅ Good win rate - strategy has edge in trends');
    }
    if (performance.sharpeRatio > 1) {
      logger.info('   ✅ Good risk-adjusted returns');
    }
    if (performance.maxDrawdownPercent < 15) {
      logger.info('   ✅ Controlled drawdown');
    }
    if (result.validSignals > 0) {
      logger.info('   ✅ Strategy generated signals in trending conditions');
    }

    // Trade analysis
    if (result.trades.length > 0) {
      logger.info('\n📋 TRADE ANALYSIS:');
      result.trades.forEach((trade: any, index: number) => {
        const duration = (trade.duration / (1000 * 60 * 60)).toFixed(1);
        logger.info(`   Trade ${index + 1}: ${trade.side} $${trade.pnl.toFixed(2)} (${trade.pnlPercent.toFixed(2)}%) - ${duration}h`);
      });
    }

    // Overall assessment
    logger.info('\n⭐ OVERALL ASSESSMENT:');
    const rating = this.getPerformanceRating(performance);
    logger.info(`   ${rating}`);

    logger.info('=' .repeat(80));
  }

  private getPerformanceRating(performance: any): string {
    let score = 0;

    if (performance.totalReturnPercent > 20) score += 2;
    else if (performance.totalReturnPercent > 10) score += 1;
    else if (performance.totalReturnPercent > 0) score += 0;
    else score -= 1;

    if (performance.sharpeRatio > 1.5) score += 2;
    else if (performance.sharpeRatio > 1) score += 1;
    else if (performance.sharpeRatio > 0) score += 0;
    else score -= 1;

    if (performance.winRate > 60) score += 1;
    else if (performance.winRate > 40) score += 0;
    else score -= 1;

    if (performance.maxDrawdownPercent < 10) score += 1;
    else if (performance.maxDrawdownPercent < 20) score += 0;
    else score -= 1;

    if (score >= 4) return '🌟 EXCELLENT - Strategy performs very well in trending markets';
    else if (score >= 2) return '✅ GOOD - Strategy shows promise in trending conditions';
    else if (score >= 0) return '⚠️ AVERAGE - Strategy needs optimization';
    else return '❌ POOR - Strategy requires significant improvements';
  }
}

/**
 * Main execution function
 */
async function main() {
  const test = new TrendingMarketTest();
  
  try {
    await test.runTest();
  } catch (error) {
    logger.error('💥 Trending market test failed:', error);
    process.exit(1);
  }
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { TrendingMarketTest };
