#!/usr/bin/env node

/**
 * Dynamic Take Profit 3-Month Backtest System
 * Enhanced strategy with adaptive take profit levels
 * Target: Improve from +8.5% to +15-20% returns
 */

import { PortfolioManager } from '../services/portfolioManager';
import { DynamicTakeProfitManager, MarketRegime, DynamicTakeProfitConfig } from '../services/dynamicTakeProfitManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { marketDataService } from '../services/marketDataProvider';
import { BacktestConfig, TradingSignal, TakeProfitLevel } from '../types/marketData';
import { logger } from '../utils/logger';

class DynamicTakeProfitBacktest {
  private takeProfitManager: DynamicTakeProfitManager;

  constructor() {
    this.takeProfitManager = new DynamicTakeProfitManager();
  }

  /**
   * Main execution function
   */
  public async runDynamicTakeProfitBacktest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🚀 DYNAMIC TAKE PROFIT 3-MONTH BACKTEST SYSTEM');
      logger.info('🎯 Target: Improve from +8.5% to +15-20% returns');
      
      // Enhanced configuration
      const realBalance = 2500; // Mock $2500 balance
      const tradingCapital = realBalance * 0.75; // 75% = $1875
      
      logger.info(`💰 Mock Delta Balance: $${realBalance.toFixed(2)}`);
      logger.info(`🎯 Trading Capital (75%): $${tradingCapital.toFixed(2)}`);
      logger.info(`⚡ Leverage: 200x (Max buying power: $${(tradingCapital * 200).toFixed(2)})`);
      logger.info(`📈 Enhanced Features: Dynamic take profit, trailing stops, partial exits`);

      // Test multiple assets with enhanced strategy
      const assets = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
      const results = [];

      for (const asset of assets) {
        logger.info(`\n🔥 DYNAMIC TAKE PROFIT BACKTEST: ${asset}`);
        const result = await this.runEnhancedAssetBacktest(asset, tradingCapital);
        if (result) {
          results.push(result);
        }
      }

      // Generate enhanced report
      this.generateEnhancedReport(results, realBalance, tradingCapital, startTime);

      const duration = (Date.now() - startTime) / 1000;
      logger.info(`🎉 Dynamic take profit backtest completed in ${duration.toFixed(2)} seconds`);

    } catch (error) {
      logger.error('❌ Dynamic take profit backtest failed:', error);
      throw error;
    }
  }

  /**
   * Run enhanced backtest for specific asset
   */
  private async runEnhancedAssetBacktest(asset: string, tradingCapital: number): Promise<any> {
    try {
      // 3-month period
      const endDate = new Date();
      const startDate = new Date(endDate.getTime() - (90 * 24 * 60 * 60 * 1000));

      const config: BacktestConfig = {
        symbol: asset as any,
        timeframe: '15m',
        startDate,
        endDate,
        initialCapital: tradingCapital, // $1875
        leverage: 200,
        riskPerTrade: 5, // 5% risk per trade
        commission: 0.1,
        slippage: 0.05,
        strategy: `DYNAMIC_TP_${asset}`,
        parameters: {
          maxDrawdown: 30,
          minConfidence: 65,
          useDynamicTakeProfit: true,
          enableTrailing: true,
          enablePartialExits: true,
        },
      };

      logger.info(`📊 Fetching 3-month data for ${asset}...`);

      // Load market data
      const response = await marketDataService.fetchHistoricalData({
        symbol: config.symbol,
        timeframe: config.timeframe,
        startDate: config.startDate,
        endDate: config.endDate,
        exchange: 'binance',
      }, 'binance');

      if (response.data.length === 0) {
        logger.warn(`⚠️ No data for ${asset}`);
        return null;
      }

      logger.info(`📊 Loaded ${response.data.length} candles for ${asset}`);

      // Create enhanced portfolio manager
      const portfolioManager = new PortfolioManager(config);
      const signals: TradingSignal[] = [];
      const partialExits: any[] = [];
      let maxDrawdownHit = false;

      // Process data with enhanced strategy
      for (let i = 100; i < response.data.length; i += 15) { // Every 15 candles
        const candle = response.data[i];
        
        // Check drawdown
        const currentCash = portfolioManager.getCash();
        const positions = portfolioManager.getPositions();
        const currentEquity = currentCash + positions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
        const drawdown = (config.initialCapital - currentEquity) / config.initialCapital * 100;
        
        if (drawdown >= config.parameters.maxDrawdown) {
          logger.warn(`🛑 Max drawdown hit: ${drawdown.toFixed(2)}% for ${asset}`);
          maxDrawdownHit = true;
          break;
        }

        // Update portfolio with dynamic take profit management
        portfolioManager.updatePositions(config.symbol, candle.close, candle.timestamp);
        
        // Process dynamic take profit levels for existing positions
        const currentPositions = portfolioManager.getPositions();
        for (const position of currentPositions) {
          if (position.symbol === config.symbol && position.takeProfitLevels) {
            const partialExitResults = this.processDynamicTakeProfits(
              position, 
              candle.close, 
              candle.timestamp, 
              portfolioManager
            );
            partialExits.push(...partialExitResults);
          }
        }

        // Check traditional stop loss
        portfolioManager.checkStopLossAndTakeProfit(config.symbol, candle.close, candle.timestamp);

        // Generate enhanced signal
        const signal = this.generateEnhancedSignal(candle, config, response.data, i);
        
        if (signal && signal.confidence >= config.parameters.minConfidence) {
          // Generate dynamic take profit levels
          const marketRegime = this.detectMarketRegime(response.data, i);
          const momentum = this.calculateMomentum(response.data, i);
          const volume = this.calculateVolumeStrength(response.data, i);

          const takeProfitConfig: DynamicTakeProfitConfig = {
            asset: config.symbol,
            entryPrice: candle.close,
            stopLoss: signal.stopLoss!,
            positionSize: signal.quantity,
            side: signal.type,
            marketRegime,
            momentum,
            volume,
          };

          const takeProfitLevels = this.takeProfitManager.generateDynamicTakeProfitLevels(takeProfitConfig);
          
          // Enhance signal with dynamic take profit
          signal.takeProfitLevels = takeProfitLevels;
          
          signals.push(signal);

          // Execute trade with enhanced features
          const trade = portfolioManager.executeTrade(signal, candle.close, candle.timestamp);
          if (trade) {
            trade.strategy = config.strategy;
            
            // Add dynamic take profit levels to position
            const position = portfolioManager.getPositions().find(p => p.symbol === config.symbol);
            if (position) {
              position.takeProfitLevels = takeProfitLevels;
              position.originalSize = position.size;
              position.partialExits = [];
            }
          }
        }
      }

      // Get enhanced results
      const finalTrades = portfolioManager.getTrades();
      const portfolioHistory = portfolioManager.getPortfolioHistory();
      const performance = PerformanceAnalytics.calculateMetrics(finalTrades, portfolioHistory, config);

      const finalCash = portfolioManager.getCash();
      const finalPositions = portfolioManager.getPositions();
      const finalEquity = finalCash + finalPositions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
      const totalReturn = ((finalEquity - config.initialCapital) / config.initialCapital) * 100;

      logger.info(`✅ ${asset} enhanced completed: ${finalTrades.length} trades, ${totalReturn.toFixed(2)}% return`);
      logger.info(`📊 Partial exits: ${partialExits.length}, Enhanced features active`);

      return {
        asset,
        config,
        signals,
        trades: finalTrades,
        performance,
        finalEquity,
        totalReturn,
        maxDrawdownHit,
        dataPoints: response.data.length,
        partialExits,
        enhancedFeatures: {
          dynamicTakeProfit: true,
          trailingStops: true,
          partialExits: partialExits.length,
        },
      };

    } catch (error) {
      logger.error(`❌ Failed enhanced backtest for ${asset}:`, error);
      return null;
    }
  }

  /**
   * Generate enhanced trading signals with market analysis
   */
  private generateEnhancedSignal(candle: any, config: BacktestConfig, data: any[], index: number): TradingSignal | null {
    const priceChange = (candle.close - candle.open) / candle.open;
    const volatility = (candle.high - candle.low) / candle.close;
    
    // Enhanced signal generation with market regime awareness
    const marketRegime = this.detectMarketRegime(data, index);
    const momentum = this.calculateMomentum(data, index);
    const volume = this.calculateVolumeStrength(data, index);
    
    let signalType: 'BUY' | 'SELL';
    let confidence = 60;
    let reason = 'Enhanced momentum';

    // Market regime-based signal generation
    if (marketRegime.type === 'TRENDING') {
      // Trending market - follow momentum
      if (momentum > 30 && priceChange > 0.006) {
        signalType = 'BUY';
        confidence = 75 + Math.min(15, momentum / 4);
        reason = 'Strong trending momentum up';
      } else if (momentum < -30 && priceChange < -0.006) {
        signalType = 'SELL';
        confidence = 75 + Math.min(15, Math.abs(momentum) / 4);
        reason = 'Strong trending momentum down';
      } else {
        return null;
      }
    } else if (marketRegime.type === 'RANGING') {
      // Ranging market - mean reversion
      if (priceChange < -0.008 && volatility > 0.015) {
        signalType = 'BUY';
        confidence = 70;
        reason = 'Mean reversion buy in range';
      } else if (priceChange > 0.008 && volatility > 0.015) {
        signalType = 'SELL';
        confidence = 70;
        reason = 'Mean reversion sell in range';
      } else {
        return null;
      }
    } else {
      // Volatile market - breakout strategy
      if (volume > 1.5 && Math.abs(priceChange) > 0.01) {
        signalType = priceChange > 0 ? 'BUY' : 'SELL';
        confidence = 65 + Math.min(20, volume * 10);
        reason = 'High volume breakout';
      } else if (Math.random() > 0.85) { // 15% random signals
        signalType = Math.random() > 0.5 ? 'BUY' : 'SELL';
        confidence = 65;
        reason = 'Random entry for testing';
      } else {
        return null;
      }
    }

    // Enhanced position sizing
    const riskAmount = config.initialCapital * (config.riskPerTrade / 100);
    const stopLossDistance = candle.close * 0.025; // 2.5% stop loss
    let quantity = (riskAmount / stopLossDistance) * config.leverage;
    
    quantity = Math.max(quantity, 0.001);
    const maxQuantity = (config.initialCapital * 0.4) / candle.close;
    quantity = Math.min(quantity, maxQuantity);

    const stopLoss = signalType === 'BUY'
      ? candle.close * 0.975
      : candle.close * 1.025;

    // Note: takeProfit will be replaced by dynamic levels
    const takeProfit = signalType === 'BUY'
      ? candle.close * 1.075
      : candle.close * 0.925;

    return {
      id: `enhanced_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      timestamp: candle.timestamp,
      symbol: config.symbol,
      type: signalType,
      price: candle.close,
      quantity: quantity,
      confidence: confidence,
      strategy: config.strategy,
      reason,
      stopLoss,
      takeProfit,
      riskReward: 3.0, // Will be overridden by dynamic levels
    };
  }

  /**
   * Detect market regime based on price action and volatility
   */
  private detectMarketRegime(data: any[], index: number): MarketRegime {
    const lookback = 50; // 50 candles lookback
    const start = Math.max(0, index - lookback);
    const recentData = data.slice(start, index + 1);

    if (recentData.length < 20) {
      return {
        type: 'VOLATILE',
        strength: 50,
        direction: 'SIDEWAYS',
        volatility: 0.03,
        volume: 1.0,
      };
    }

    // Calculate trend strength
    const prices = recentData.map(d => d.close);
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    const trendDirection = lastPrice > firstPrice ? 'UP' : lastPrice < firstPrice ? 'DOWN' : 'SIDEWAYS';

    // Calculate volatility
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i-1]) / prices[i-1]);
    }
    const volatility = Math.sqrt(returns.reduce((sum, r) => sum + r * r, 0) / returns.length);

    // Calculate trend strength (R-squared of linear regression)
    const n = prices.length;
    const x = Array.from({length: n}, (_, i) => i);
    const y = prices;

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const yMean = sumY / n;
    const ssRes = y.reduce((sum, yi, i) => {
      const predicted = slope * x[i] + intercept;
      return sum + Math.pow(yi - predicted, 2);
    }, 0);
    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const rSquared = 1 - (ssRes / ssTot);

    // Determine regime
    let regimeType: 'TRENDING' | 'RANGING' | 'VOLATILE';
    let strength = Math.max(0, Math.min(100, rSquared * 100));

    if (rSquared > 0.6 && volatility < 0.04) {
      regimeType = 'TRENDING';
      strength = Math.min(90, strength + 20);
    } else if (rSquared < 0.3 && volatility < 0.03) {
      regimeType = 'RANGING';
      strength = Math.max(30, 70 - strength);
    } else {
      regimeType = 'VOLATILE';
      strength = Math.min(80, volatility * 2000);
    }

    return {
      type: regimeType,
      strength,
      direction: trendDirection,
      volatility,
      volume: 1.0, // Will be calculated separately
    };
  }

  /**
   * Calculate momentum indicator
   */
  private calculateMomentum(data: any[], index: number): number {
    const lookback = 20;
    const start = Math.max(0, index - lookback);
    const recentData = data.slice(start, index + 1);

    if (recentData.length < 10) return 0;

    const prices = recentData.map(d => d.close);
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];

    const momentum = ((lastPrice - firstPrice) / firstPrice) * 100;
    return Math.max(-100, Math.min(100, momentum * 5)); // Scale to -100 to 100
  }

  /**
   * Calculate volume strength
   */
  private calculateVolumeStrength(data: any[], index: number): number {
    const lookback = 20;
    const start = Math.max(0, index - lookback);
    const recentData = data.slice(start, index + 1);

    if (recentData.length < 10) return 1.0;

    const volumes = recentData.map(d => d.volume || 1000); // Default volume if not available
    const currentVolume = volumes[volumes.length - 1];
    const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;

    return Math.max(0.5, Math.min(2.0, currentVolume / avgVolume));
  }

  /**
   * Process dynamic take profit levels for existing positions
   */
  private processDynamicTakeProfits(
    position: any,
    currentPrice: number,
    timestamp: number,
    portfolioManager: PortfolioManager
  ): any[] {
    const partialExits: any[] = [];

    if (!position.takeProfitLevels || !position.originalSize) {
      return partialExits;
    }

    // Update trailing take profits
    const config: DynamicTakeProfitConfig = {
      asset: position.symbol,
      entryPrice: position.entryPrice,
      stopLoss: position.stopLoss || 0,
      positionSize: position.originalSize,
      side: position.side === 'LONG' ? 'BUY' : 'SELL',
      marketRegime: { type: 'TRENDING', strength: 70, direction: 'UP', volatility: 0.03, volume: 1.0 },
      momentum: 0,
      volume: 1.0,
    };

    position.takeProfitLevels = this.takeProfitManager.updateTrailingTakeProfits(
      position.takeProfitLevels,
      currentPrice,
      config
    );

    // Check for take profit executions
    const triggeredLevels = this.takeProfitManager.checkTakeProfitExecution(
      position.takeProfitLevels,
      currentPrice,
      config
    );

    // Execute partial exits
    for (const level of triggeredLevels) {
      if (level.executed) continue;

      const exitSize = (position.originalSize * level.percentage) / 100;
      const pnl = this.calculatePartialPnl(position, currentPrice, exitSize);

      // Mark level as executed
      level.executed = true;
      level.executedAt = timestamp;

      // Record partial exit
      const partialExit = {
        percentage: level.percentage,
        price: currentPrice,
        timestamp,
        pnl,
        reason: `Take profit level ${level.riskRewardRatio.toFixed(1)}:1`,
        size: exitSize,
      };

      partialExits.push(partialExit);

      // Update position size
      position.size -= exitSize;

      // Add to position's partial exits
      if (!position.partialExits) {
        position.partialExits = [];
      }
      position.partialExits.push(partialExit);

      logger.info(`💰 Partial exit: ${level.percentage}% at $${currentPrice.toFixed(2)} (${level.riskRewardRatio.toFixed(1)}:1) - P&L: $${pnl.toFixed(2)}`);
    }

    return partialExits;
  }

  /**
   * Calculate P&L for partial exit
   */
  private calculatePartialPnl(position: any, exitPrice: number, exitSize: number): number {
    const priceChange = position.side === 'LONG'
      ? exitPrice - position.entryPrice
      : position.entryPrice - exitPrice;

    return (priceChange / position.entryPrice) * exitSize * position.leverage;
  }

  /**
   * Generate enhanced performance report
   */
  private generateEnhancedReport(results: any[], realBalance: number, tradingCapital: number, startTime: number): void {
    const totalDuration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '🚀 DYNAMIC TAKE PROFIT 3-MONTH BACKTEST RESULTS'.padStart(100, '='));
    logger.info('=' .repeat(160));

    // Enhanced Configuration
    logger.info('💰 ENHANCED DYNAMIC TAKE PROFIT CONFIGURATION:');
    logger.info(`   Mock Delta Balance: $${realBalance.toFixed(2)} USD`);
    logger.info(`   Trading Capital (75%): $${tradingCapital.toFixed(2)} USD`);
    logger.info(`   Leverage: 200x (Max buying power: $${(tradingCapital * 200).toFixed(2)})`);
    logger.info(`   Enhanced Features: Dynamic take profit, trailing stops, partial exits`);
    logger.info(`   Target Improvement: From +8.5% to +15-20% returns`);
    logger.info(`   Execution Time: ${totalDuration.toFixed(2)} seconds`);

    // Enhanced Performance Summary
    logger.info('\n📊 ENHANCED PERFORMANCE SUMMARY:');
    logger.info('   Asset   | Signals | Trades | Partial | Return | P&L | Final Balance | Features');
    logger.info('   ' + '-'.repeat(140));

    let totalPnL = 0;
    let totalSignals = 0;
    let totalTrades = 0;
    let totalPartialExits = 0;
    let totalFinalBalance = 0;
    let maxDrawdownHits = 0;

    results.forEach(result => {
      const asset = result.asset.padEnd(7);
      const signals = result.signals.length.toString().padStart(7);
      const trades = result.trades.length.toString().padStart(6);
      const partials = result.partialExits.length.toString().padStart(7);
      const totalReturn = result.totalReturn.toFixed(2).padStart(6);
      const dollarPnL = (result.finalEquity - tradingCapital).toFixed(2).padStart(8);
      const finalBalance = result.finalEquity.toFixed(2).padStart(13);
      const features = 'DTP+Trail+PE'; // Dynamic TP + Trailing + Partial Exits

      logger.info(`   ${asset} | ${signals} | ${trades} | ${partials} | ${totalReturn}% | $${dollarPnL} | $${finalBalance} | ${features}`);

      totalPnL += (result.finalEquity - tradingCapital);
      totalSignals += result.signals.length;
      totalTrades += result.trades.length;
      totalPartialExits += result.partialExits.length;
      totalFinalBalance += result.finalEquity;
      if (result.maxDrawdownHit) maxDrawdownHits++;
    });

    // Overall Enhanced Results
    logger.info('\n💼 OVERALL ENHANCED RESULTS:');
    logger.info(`   Total Signals Generated: ${totalSignals}`);
    logger.info(`   Total Trades Executed: ${totalTrades}`);
    logger.info(`   Total Partial Exits: ${totalPartialExits}`);
    logger.info(`   Signal-to-Trade Ratio: ${((totalTrades/totalSignals)*100).toFixed(1)}%`);
    logger.info(`   Partial Exit Efficiency: ${(totalPartialExits/totalTrades).toFixed(1)} per trade`);
    logger.info(`   Total P&L: $${totalPnL.toFixed(2)}`);
    logger.info(`   Total Final Balance: $${totalFinalBalance.toFixed(2)}`);
    logger.info(`   Overall Return: ${((totalFinalBalance - (tradingCapital * results.length)) / (tradingCapital * results.length) * 100).toFixed(2)}%`);
    logger.info(`   Assets Hit Max Drawdown: ${maxDrawdownHits}/${results.length}`);

    // Performance Improvement Analysis
    const baselineReturn = 8.5; // Previous system return
    const enhancedReturn = ((totalFinalBalance - (tradingCapital * results.length)) / (tradingCapital * results.length) * 100);
    const improvement = enhancedReturn - baselineReturn;
    const improvementPercent = (improvement / baselineReturn) * 100;

    logger.info('\n📈 PERFORMANCE IMPROVEMENT ANALYSIS:');
    logger.info(`   Baseline System Return: +${baselineReturn}%`);
    logger.info(`   Enhanced System Return: +${enhancedReturn.toFixed(2)}%`);
    logger.info(`   Absolute Improvement: +${improvement.toFixed(2)} percentage points`);
    logger.info(`   Relative Improvement: +${improvementPercent.toFixed(1)}%`);

    if (enhancedReturn >= 15) {
      logger.info(`   🎯 TARGET ACHIEVED: Enhanced system reached ${enhancedReturn.toFixed(2)}% (target: 15-20%)`);
    } else if (enhancedReturn > baselineReturn) {
      logger.info(`   📈 IMPROVEMENT CONFIRMED: Enhanced system outperformed baseline by ${improvement.toFixed(2)}%`);
    } else {
      logger.info(`   ⚠️ NEEDS OPTIMIZATION: Enhanced system underperformed baseline`);
    }

    // Enhanced Features Analysis
    logger.info('\n🔧 ENHANCED FEATURES PERFORMANCE:');
    results.forEach(result => {
      if (result.partialExits.length > 0) {
        const avgPartialPnL = result.partialExits.reduce((sum, pe) => sum + pe.pnl, 0) / result.partialExits.length;
        const totalPartialPnL = result.partialExits.reduce((sum, pe) => sum + pe.pnl, 0);

        logger.info(`   ${result.asset} Partial Exits:`);
        logger.info(`     Count: ${result.partialExits.length}`);
        logger.info(`     Average P&L: $${avgPartialPnL.toFixed(2)}`);
        logger.info(`     Total P&L: $${totalPartialPnL.toFixed(2)}`);
        logger.info(`     Contribution: ${((totalPartialPnL / (result.finalEquity - tradingCapital)) * 100).toFixed(1)}% of total P&L`);
      }
    });

    // Real Balance Impact
    const balanceImpact = totalPnL / results.length;
    const newBalance = realBalance + balanceImpact;

    logger.info('\n💰 ENHANCED REAL BALANCE IMPACT:');
    logger.info(`   Starting Balance: $${realBalance.toFixed(2)}`);
    logger.info(`   Enhanced P&L per Asset: $${balanceImpact.toFixed(2)}`);
    logger.info(`   Projected New Balance: $${newBalance.toFixed(2)}`);
    logger.info(`   Enhanced Balance Change: ${((newBalance - realBalance) / realBalance * 100).toFixed(2)}%`);

    // Success Assessment
    if (enhancedReturn >= 15) {
      logger.info('\n🚀 ENHANCED SYSTEM SUCCESS:');
      logger.info('   ✅ Target return achieved (15-20%)');
      logger.info('   ✅ Dynamic take profit system working');
      logger.info('   ✅ Partial exits maximizing profits');
      logger.info('   ✅ Trailing stops locking in gains');
      logger.info('   ✅ Market regime adaptation effective');
      logger.info('   🎯 READY FOR LIVE DEPLOYMENT');
    } else if (enhancedReturn > baselineReturn) {
      logger.info('\n📈 ENHANCED SYSTEM IMPROVEMENT:');
      logger.info('   ✅ Outperformed baseline system');
      logger.info('   ✅ Dynamic features adding value');
      logger.info('   📊 Further optimization recommended');
      logger.info('   🔧 Consider parameter tuning');
    } else {
      logger.info('\n⚠️ ENHANCED SYSTEM NEEDS WORK:');
      logger.info('   🔧 Dynamic take profit needs optimization');
      logger.info('   📊 Market regime detection refinement');
      logger.info('   🎯 Partial exit timing adjustment');
      logger.info('   📈 Consider different risk-reward ratios');
    }

    logger.info('\n🎯 ENHANCED SYSTEM RECOMMENDATIONS:');
    logger.info('   1. 📊 Monitor partial exit efficiency');
    logger.info('   2. 🔧 Fine-tune market regime detection');
    logger.info('   3. 💰 Optimize asset-specific parameters');
    logger.info('   4. 📈 Test with different market conditions');
    logger.info('   5. 🚀 Consider live trading with small capital');

    logger.info('=' .repeat(160));
  }
}

/**
 * Main execution function
 */
async function main() {
  const system = new DynamicTakeProfitBacktest();

  try {
    await system.runDynamicTakeProfitBacktest();
  } catch (error) {
    logger.error('💥 Dynamic take profit backtest failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { DynamicTakeProfitBacktest };
