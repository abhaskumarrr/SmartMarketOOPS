/**
 * Moving Average Crossover Trading Strategy
 * Generates buy/sell signals based on moving average crossovers
 */

import { 
  TradingStrategy, 
  TradingSignal, 
  EnhancedMarketData, 
  BacktestConfig 
} from '../types/marketData';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';

export interface MACrossoverParams {
  fastPeriod: number;
  slowPeriod: number;
  rsiPeriod: number;
  rsiOverbought: number;
  rsiOversold: number;
  volumeThreshold: number; // Minimum volume multiplier
  stopLossPercent: number;
  takeProfitPercent: number;
  minConfidence: number;
}

export class MovingAverageCrossoverStrategy implements TradingStrategy {
  public readonly name = 'MA_Crossover';
  public parameters: MACrossoverParams;
  private config?: BacktestConfig;

  constructor(parameters?: Partial<MACrossoverParams>) {
    this.parameters = {
      fastPeriod: 20,
      slowPeriod: 50,
      rsiPeriod: 14,
      rsiOverbought: 70,
      rsiOversold: 30,
      volumeThreshold: 1.2, // 20% above average
      stopLossPercent: 2.0,
      takeProfitPercent: 4.0,
      minConfidence: 60,
      ...parameters,
    };
  }

  public initialize(config: BacktestConfig): void {
    this.config = config;
    logger.info(`🎯 Initialized ${this.name} strategy`, {
      parameters: this.parameters,
      symbol: config.symbol,
      timeframe: config.timeframe,
    });
  }

  public generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    if (!this.config) {
      throw new Error('Strategy not initialized. Call initialize() first.');
    }

    // Need enough data for indicators
    if (currentIndex < Math.max(this.parameters.fastPeriod, this.parameters.slowPeriod, this.parameters.rsiPeriod)) {
      return null;
    }

    const currentCandle = data[currentIndex];
    const previousCandle = data[currentIndex - 1];

    // Calculate indicators for the current and previous candles
    const currentIndicators = currentCandle.indicators;
    const previousIndicators = previousCandle.indicators;

    // Check if we have all required indicators
    if (!this.hasRequiredIndicators(currentIndicators) || !this.hasRequiredIndicators(previousIndicators)) {
      return null;
    }

    // Detect crossover signals
    const signal = this.detectCrossover(currentIndicators, previousIndicators);
    
    if (!signal) {
      return null;
    }

    // Apply additional filters
    const filteredSignal = this.applyFilters(signal, currentCandle, data, currentIndex);
    
    if (!filteredSignal) {
      return null;
    }

    // Calculate confidence and risk management
    const enhancedSignal = this.enhanceSignal(filteredSignal, currentCandle, data, currentIndex);

    logger.debug(`📊 Generated ${enhancedSignal.type} signal for ${currentCandle.symbol}`, {
      price: enhancedSignal.price,
      confidence: enhancedSignal.confidence,
      reason: enhancedSignal.reason,
    });

    return enhancedSignal;
  }

  private hasRequiredIndicators(indicators: any): boolean {
    return !!(
      indicators.sma_20 !== undefined && !isNaN(indicators.sma_20) &&
      indicators.sma_50 !== undefined && !isNaN(indicators.sma_50) &&
      indicators.rsi !== undefined && !isNaN(indicators.rsi) &&
      indicators.volume_sma !== undefined && !isNaN(indicators.volume_sma)
    );
  }

  private detectCrossover(current: any, previous: any): TradingSignal | null {
    const currentFastAboveSlow = current.sma_20 > current.sma_50;
    const previousFastAboveSlow = previous.sma_20 > previous.sma_50;

    // Bullish crossover: fast MA crosses above slow MA
    if (!previousFastAboveSlow && currentFastAboveSlow) {
      return {
        id: createEventId(),
        timestamp: Date.now(),
        symbol: this.config!.symbol,
        type: 'BUY',
        price: current.close || 0,
        quantity: 0, // Will be calculated later
        confidence: 50, // Base confidence
        strategy: this.name,
        reason: 'Bullish MA crossover',
      };
    }

    // Bearish crossover: fast MA crosses below slow MA
    if (previousFastAboveSlow && !currentFastAboveSlow) {
      return {
        id: createEventId(),
        timestamp: Date.now(),
        symbol: this.config!.symbol,
        type: 'SELL',
        price: current.close || 0,
        quantity: 0, // Will be calculated later
        confidence: 50, // Base confidence
        strategy: this.name,
        reason: 'Bearish MA crossover',
      };
    }

    return null;
  }

  private applyFilters(signal: TradingSignal, currentCandle: EnhancedMarketData, data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    const indicators = currentCandle.indicators;

    // RSI Filter
    if (signal.type === 'BUY' && indicators.rsi! > this.parameters.rsiOverbought) {
      logger.debug(`🚫 BUY signal filtered out: RSI overbought (${indicators.rsi})`);
      return null;
    }

    if (signal.type === 'SELL' && indicators.rsi! < this.parameters.rsiOversold) {
      logger.debug(`🚫 SELL signal filtered out: RSI oversold (${indicators.rsi})`);
      return null;
    }

    // Volume Filter
    const volumeRatio = currentCandle.volume / indicators.volume_sma!;
    if (volumeRatio < this.parameters.volumeThreshold) {
      logger.debug(`🚫 Signal filtered out: Low volume (${volumeRatio.toFixed(2)}x average)`);
      return null;
    }

    // Trend Filter (price should be above/below both MAs for stronger signals)
    if (signal.type === 'BUY' && currentCandle.close < Math.min(indicators.sma_20!, indicators.sma_50!)) {
      logger.debug(`🚫 BUY signal filtered out: Price below MAs`);
      return null;
    }

    if (signal.type === 'SELL' && currentCandle.close > Math.max(indicators.sma_20!, indicators.sma_50!)) {
      logger.debug(`🚫 SELL signal filtered out: Price above MAs`);
      return null;
    }

    return signal;
  }

  private enhanceSignal(signal: TradingSignal, currentCandle: EnhancedMarketData, data: EnhancedMarketData[], currentIndex: number): TradingSignal {
    const indicators = currentCandle.indicators;
    let confidence = signal.confidence;

    // Enhance confidence based on multiple factors
    
    // 1. RSI confirmation
    if (signal.type === 'BUY' && indicators.rsi! < 50) {
      confidence += 10; // RSI not overbought
    }
    if (signal.type === 'SELL' && indicators.rsi! > 50) {
      confidence += 10; // RSI not oversold
    }

    // 2. Volume confirmation
    const volumeRatio = currentCandle.volume / indicators.volume_sma!;
    if (volumeRatio > 1.5) {
      confidence += 15; // Strong volume
    } else if (volumeRatio > 1.2) {
      confidence += 10; // Good volume
    }

    // 3. MACD confirmation
    if (indicators.macd !== undefined && indicators.macd_signal !== undefined) {
      const macdBullish = indicators.macd > indicators.macd_signal;
      if ((signal.type === 'BUY' && macdBullish) || (signal.type === 'SELL' && !macdBullish)) {
        confidence += 10;
      }
    }

    // 4. Trend strength
    const maDiff = Math.abs(indicators.sma_20! - indicators.sma_50!) / indicators.sma_50!;
    if (maDiff > 0.02) { // 2% difference
      confidence += 10;
    }

    // 5. Price position relative to Bollinger Bands
    if (indicators.bollinger_upper && indicators.bollinger_lower) {
      const bbPosition = (currentCandle.close - indicators.bollinger_lower) / 
                        (indicators.bollinger_upper - indicators.bollinger_lower);
      
      if (signal.type === 'BUY' && bbPosition < 0.3) {
        confidence += 10; // Near lower band
      }
      if (signal.type === 'SELL' && bbPosition > 0.7) {
        confidence += 10; // Near upper band
      }
    }

    // Cap confidence at 100
    confidence = Math.min(confidence, 100);

    // Filter out low confidence signals
    if (confidence < this.parameters.minConfidence) {
      return { ...signal, confidence: 0 }; // Mark as invalid
    }

    // Calculate position size based on risk management
    const quantity = this.calculatePositionSize(currentCandle.close, confidence);

    // Calculate stop loss and take profit
    const stopLoss = signal.type === 'BUY' 
      ? currentCandle.close * (1 - this.parameters.stopLossPercent / 100)
      : currentCandle.close * (1 + this.parameters.stopLossPercent / 100);

    const takeProfit = signal.type === 'BUY'
      ? currentCandle.close * (1 + this.parameters.takeProfitPercent / 100)
      : currentCandle.close * (1 - this.parameters.takeProfitPercent / 100);

    const riskReward = this.parameters.takeProfitPercent / this.parameters.stopLossPercent;

    return {
      ...signal,
      quantity,
      confidence,
      stopLoss,
      takeProfit,
      riskReward,
      reason: `${signal.reason} (Confidence: ${confidence}%, Volume: ${volumeRatio.toFixed(2)}x, RSI: ${indicators.rsi!.toFixed(1)})`,
    };
  }

  private calculatePositionSize(price: number, confidence: number): number {
    if (!this.config) return 0;

    // Base position size on risk per trade
    const riskAmount = this.config.initialCapital * (this.config.riskPerTrade / 100);
    const stopLossDistance = price * (this.parameters.stopLossPercent / 100);
    
    // Calculate base position size
    let positionSize = riskAmount / stopLossDistance;
    
    // Apply leverage
    positionSize *= this.config.leverage;
    
    // Scale by confidence (50-100% confidence maps to 50-100% of calculated size)
    const confidenceMultiplier = 0.5 + (confidence - this.parameters.minConfidence) / 
                                 (100 - this.parameters.minConfidence) * 0.5;
    
    positionSize *= confidenceMultiplier;
    
    // Ensure minimum position size
    return Math.max(positionSize, 0.001);
  }

  public getDescription(): string {
    return `Moving Average Crossover Strategy (${this.parameters.fastPeriod}/${this.parameters.slowPeriod}) with RSI filter and volume confirmation`;
  }

  public getParameters(): MACrossoverParams {
    return { ...this.parameters };
  }

  public updateParameters(newParams: Partial<MACrossoverParams>): void {
    this.parameters = { ...this.parameters, ...newParams };
    logger.info(`📊 Updated ${this.name} strategy parameters`, this.parameters);
  }
}

// Export factory function for easy instantiation
export function createMACrossoverStrategy(params?: Partial<MACrossoverParams>): MovingAverageCrossoverStrategy {
  return new MovingAverageCrossoverStrategy(params);
}
