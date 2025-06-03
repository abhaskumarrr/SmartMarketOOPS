#!/usr/bin/env node

/**
 * Fast 3-Month Backtest System
 * Streamlined version for quick execution
 * Mock Delta balance: $2500, 75% usage = $1875, 200x leverage
 */

import { PortfolioManager } from '../services/portfolioManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { marketDataService } from '../services/marketDataProvider';
import { BacktestConfig, TradingSignal } from '../types/marketData';
import { logger } from '../utils/logger';

class Fast3MonthBacktest {

  /**
   * Main execution function
   */
  public async runFast3MonthBacktest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🚀 FAST 3-MONTH BACKTEST SYSTEM');
      
      // Mock Delta balance
      const realBalance = 2500; // Mock $2500 balance
      const tradingCapital = realBalance * 0.75; // 75% = $1875
      
      logger.info(`💰 Mock Delta Balance: $${realBalance.toFixed(2)}`);
      logger.info(`🎯 Trading Capital (75%): $${tradingCapital.toFixed(2)}`);
      logger.info(`⚡ Leverage: 200x (Max buying power: $${(tradingCapital * 200).toFixed(2)})`);
      logger.info(`📅 Period: 3 months backtest on BTC, ETH, SOL`);

      // Test multiple assets
      const assets = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
      const results = [];

      for (const asset of assets) {
        logger.info(`\n🔥 3-MONTH BACKTEST: ${asset}`);
        const result = await this.runAssetBacktest(asset, tradingCapital);
        if (result) {
          results.push(result);
        }
      }

      // Generate report
      this.generateFastReport(results, realBalance, tradingCapital, startTime);

      const duration = (Date.now() - startTime) / 1000;
      logger.info(`🎉 Fast 3-month backtest completed in ${duration.toFixed(2)} seconds`);

    } catch (error) {
      logger.error('❌ Fast 3-month backtest failed:', error);
      throw error;
    }
  }

  /**
   * Run backtest for specific asset
   */
  private async runAssetBacktest(asset: string, tradingCapital: number): Promise<any> {
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
        strategy: `FAST_3M_${asset}`,
        parameters: {
          maxDrawdown: 30,
          minConfidence: 65,
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

      // Create portfolio manager
      const portfolioManager = new PortfolioManager(config);
      const signals: TradingSignal[] = [];
      let maxDrawdownHit = false;

      // Process data (every 20 candles for speed)
      for (let i = 100; i < response.data.length; i += 20) {
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

        // Update portfolio
        portfolioManager.updatePositions(config.symbol, candle.close, candle.timestamp);
        portfolioManager.checkStopLossAndTakeProfit(config.symbol, candle.close, candle.timestamp);

        // Generate signal
        const signal = this.generateFastSignal(candle, config);
        
        if (signal && signal.confidence >= config.parameters.minConfidence) {
          signals.push(signal);

          // Execute trade
          const trade = portfolioManager.executeTrade(signal, candle.close, candle.timestamp);
          if (trade) {
            trade.strategy = config.strategy;
          }
        }
      }

      // Get results
      const finalTrades = portfolioManager.getTrades();
      const portfolioHistory = portfolioManager.getPortfolioHistory();
      const performance = PerformanceAnalytics.calculateMetrics(finalTrades, portfolioHistory, config);

      const finalCash = portfolioManager.getCash();
      const finalPositions = portfolioManager.getPositions();
      const finalEquity = finalCash + finalPositions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
      const totalReturn = ((finalEquity - config.initialCapital) / config.initialCapital) * 100;

      logger.info(`✅ ${asset} completed: ${finalTrades.length} trades, ${totalReturn.toFixed(2)}% return`);

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
      };

    } catch (error) {
      logger.error(`❌ Failed ${asset}:`, error);
      return null;
    }
  }

  /**
   * Generate fast trading signals
   */
  private generateFastSignal(candle: any, config: BacktestConfig): TradingSignal | null {
    const priceChange = (candle.close - candle.open) / candle.open;
    const volatility = (candle.high - candle.low) / candle.close;
    
    let signalType: 'BUY' | 'SELL';
    let confidence = 60;
    let reason = 'Fast momentum';

    // Strong momentum signals
    if (priceChange > 0.008 && volatility > 0.015) {
      signalType = 'BUY';
      confidence = 75;
      reason = 'Strong bullish momentum';
    } else if (priceChange < -0.008 && volatility > 0.015) {
      signalType = 'SELL';
      confidence = 75;
      reason = 'Strong bearish momentum';
    } else if (Math.random() > 0.8) { // 20% random signals
      signalType = Math.random() > 0.5 ? 'BUY' : 'SELL';
      confidence = 65;
      reason = 'Random entry';
    } else {
      return null;
    }

    // Position sizing for 200x leverage
    const riskAmount = config.initialCapital * (config.riskPerTrade / 100); // 5% of $1875 = $93.75
    const stopLossDistance = candle.close * 0.03; // 3% stop loss
    let quantity = (riskAmount / stopLossDistance) * config.leverage;
    
    quantity = Math.max(quantity, 0.001);
    const maxQuantity = (config.initialCapital * 0.5) / candle.close; // Max 50% of capital
    quantity = Math.min(quantity, maxQuantity);

    const stopLoss = signalType === 'BUY'
      ? candle.close * 0.97   // 3% below
      : candle.close * 1.03;  // 3% above

    const takeProfit = signalType === 'BUY'
      ? candle.close * 1.09   // 9% above (3:1 ratio)
      : candle.close * 0.91;  // 9% below

    return {
      id: `fast_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
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
      riskReward: 3.0,
    };
  }

  /**
   * Generate fast report
   */
  private generateFastReport(results: any[], realBalance: number, tradingCapital: number, startTime: number): void {
    const totalDuration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '💰 FAST 3-MONTH BACKTEST RESULTS'.padStart(80, '='));
    logger.info('=' .repeat(120));

    logger.info('💰 CONFIGURATION:');
    logger.info(`   Mock Delta Balance: $${realBalance.toFixed(2)}`);
    logger.info(`   Trading Capital (75%): $${tradingCapital.toFixed(2)}`);
    logger.info(`   Leverage: 200x (Max: $${(tradingCapital * 200).toFixed(2)})`);
    logger.info(`   Risk Per Trade: 5% ($${(tradingCapital * 0.05).toFixed(2)})`);
    logger.info(`   Execution Time: ${totalDuration.toFixed(2)} seconds`);

    logger.info('\n📊 PERFORMANCE SUMMARY:');
    logger.info('   Asset   | Signals | Trades | Return | P&L | Final Balance | Max DD');
    logger.info('   ' + '-'.repeat(80));

    let totalPnL = 0;
    let totalSignals = 0;
    let totalTrades = 0;
    let totalFinalBalance = 0;
    let maxDrawdownHits = 0;

    results.forEach(result => {
      const asset = result.asset.padEnd(7);
      const signals = result.signals.length.toString().padStart(7);
      const trades = result.trades.length.toString().padStart(6);
      const totalReturn = result.totalReturn.toFixed(2).padStart(6);
      const dollarPnL = (result.finalEquity - tradingCapital).toFixed(2).padStart(8);
      const finalBalance = result.finalEquity.toFixed(2).padStart(13);
      const maxDD = result.maxDrawdownHit ? 'YES' : 'NO';

      logger.info(`   ${asset} | ${signals} | ${trades} | ${totalReturn}% | $${dollarPnL} | $${finalBalance} | ${maxDD}`);

      totalPnL += (result.finalEquity - tradingCapital);
      totalSignals += result.signals.length;
      totalTrades += result.trades.length;
      totalFinalBalance += result.finalEquity;
      if (result.maxDrawdownHit) maxDrawdownHits++;
    });

    logger.info('\n💼 OVERALL RESULTS:');
    logger.info(`   Total Signals: ${totalSignals}`);
    logger.info(`   Total Trades: ${totalTrades}`);
    logger.info(`   Execution Rate: ${((totalTrades/totalSignals)*100).toFixed(1)}%`);
    logger.info(`   Total P&L: $${totalPnL.toFixed(2)}`);
    logger.info(`   Final Balance: $${totalFinalBalance.toFixed(2)}`);
    logger.info(`   Overall Return: ${((totalFinalBalance - (tradingCapital * results.length)) / (tradingCapital * results.length) * 100).toFixed(2)}%`);
    logger.info(`   Drawdown Hits: ${maxDrawdownHits}/${results.length}`);

    // Real balance impact
    const balanceImpact = totalPnL / results.length;
    const newBalance = realBalance + balanceImpact;
    
    logger.info('\n💰 REAL BALANCE IMPACT:');
    logger.info(`   Starting Balance: $${realBalance.toFixed(2)}`);
    logger.info(`   Projected New Balance: $${newBalance.toFixed(2)}`);
    logger.info(`   Balance Change: ${((newBalance - realBalance) / realBalance * 100).toFixed(2)}%`);

    // Trade analysis
    logger.info('\n📈 TRADE ANALYSIS:');
    results.forEach(result => {
      if (result.trades.length > 0) {
        const wins = result.trades.filter(t => t.pnl > 0);
        const losses = result.trades.filter(t => t.pnl <= 0);
        const winRate = (wins.length / result.trades.length * 100).toFixed(1);
        const avgWin = wins.length > 0 ? (wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length).toFixed(2) : '0.00';
        const avgLoss = losses.length > 0 ? (losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length).toFixed(2) : '0.00';
        
        logger.info(`   ${result.asset}: ${result.trades.length} trades, ${winRate}% win rate, Avg Win: $${avgWin}, Avg Loss: $${avgLoss}`);
      }
    });

    if (totalPnL > 0) {
      logger.info('\n🚀 SUCCESS: Profitable 3-month backtest!');
      logger.info(`   Generated $${totalPnL.toFixed(2)} profit with 200x leverage`);
      logger.info(`   ROI: ${((totalPnL / (tradingCapital * results.length)) * 100).toFixed(2)}%`);
      logger.info(`   Annualized: ${(((totalPnL / (tradingCapital * results.length)) * 4) * 100).toFixed(2)}%`);
    } else {
      logger.info('\n⚠️ LOSS: Strategy needs optimization');
      logger.info(`   Lost $${Math.abs(totalPnL).toFixed(2)} over 3 months`);
    }

    logger.info('\n🎯 READY FOR LIVE TRADING CONSIDERATION');
    logger.info('=' .repeat(120));
  }
}

/**
 * Main execution
 */
async function main() {
  const system = new Fast3MonthBacktest();
  
  try {
    await system.runFast3MonthBacktest();
  } catch (error) {
    logger.error('💥 Fast backtest failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { Fast3MonthBacktest };
