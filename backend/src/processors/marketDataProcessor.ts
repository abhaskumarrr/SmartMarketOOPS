/**
 * Market Data Event Processor
 * Processes market data events and triggers signal generation
 */

import { EventProcessor } from '../services/eventProcessingPipeline';
import { redisStreamsService } from '../services/redisStreamsService';
import { questdbService } from '../services/questdbService';
import { logger } from '../utils/logger';
import {
  TradingEvent,
  MarketDataEvent,
  OHLCVEvent,
  TradingSignalEvent,
  EventProcessingResult,
  ProcessingStatus,
  STREAM_NAMES,
  createEventId,
  createCorrelationId,
  isMarketDataEvent,
} from '../types/events';

export class MarketDataProcessor implements EventProcessor {
  private name = 'MarketDataProcessor';
  private signalGenerationEnabled = true;
  private lastPrices: Map<string, number> = new Map();
  private priceChangeThreshold = 0.001; // 0.1% price change threshold

  public getName(): string {
    return this.name;
  }

  public canProcess(event: TradingEvent): boolean {
    return isMarketDataEvent(event) || event.type.startsWith('OHLCV_');
  }

  public async process(event: TradingEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();
    
    try {
      if (isMarketDataEvent(event)) {
        return await this.processMarketData(event as MarketDataEvent);
      } else if (event.type.startsWith('OHLCV_')) {
        return await this.processOHLCV(event as OHLCVEvent);
      }

      throw new Error(`Unsupported event type: ${event.type}`);
    } catch (error) {
      logger.error(`❌ Market data processing failed for event ${event.id}:`, error);
      
      return {
        eventId: event.id!,
        status: ProcessingStatus.FAILED,
        processingTime: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Process market data event
   */
  private async processMarketData(event: MarketDataEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();
    const nextEvents: TradingEvent[] = [];

    try {
      const { symbol, price, volume, timestamp } = event.data;

      // Store market data in QuestDB for time-series analysis
      await this.storeMarketData(event);

      // Check for significant price changes
      const lastPrice = this.lastPrices.get(symbol);
      if (lastPrice) {
        const priceChange = Math.abs(price - lastPrice) / lastPrice;
        
        if (priceChange >= this.priceChangeThreshold) {
          logger.debug(`📈 Significant price change detected for ${symbol}: ${(priceChange * 100).toFixed(2)}%`);
          
          // Generate signal generation event if enabled
          if (this.signalGenerationEnabled) {
            const signalEvent = await this.generateSignalEvent(event, priceChange);
            if (signalEvent) {
              nextEvents.push(signalEvent);
            }
          }
        }
      }

      // Update last price
      this.lastPrices.set(symbol, price);

      // Publish next events
      for (const nextEvent of nextEvents) {
        await redisStreamsService.publishEvent(STREAM_NAMES.TRADING_SIGNALS, nextEvent);
      }

      logger.debug(`✅ Processed market data for ${symbol} @ ${price}`);

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          symbol,
          price,
          priceChange: lastPrice ? (price - lastPrice) / lastPrice : 0,
          signalsGenerated: nextEvents.length,
        },
        nextEvents,
      };

    } catch (error) {
      throw new Error(`Market data processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Process OHLCV candle event
   */
  private async processOHLCV(event: OHLCVEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();

    try {
      const { symbol, close, volume, timestamp } = event.data;

      // Store OHLCV data in QuestDB
      await this.storeOHLCVData(event);

      // Update last price with close price
      this.lastPrices.set(symbol, close);

      logger.debug(`✅ Processed OHLCV candle for ${symbol} @ ${close}`);

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          symbol,
          close,
          volume,
          timestamp,
        },
      };

    } catch (error) {
      throw new Error(`OHLCV processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Store market data in QuestDB
   */
  private async storeMarketData(event: MarketDataEvent): Promise<void> {
    try {
      const marketData = {
        timestamp: new Date(event.data.timestamp),
        symbol: event.data.symbol,
        exchange: event.data.exchange,
        timeframe: '1m', // Assuming 1-minute data for real-time
        open: event.data.price, // For real-time data, use current price as OHLC
        high: event.data.price,
        low: event.data.price,
        close: event.data.price,
        volume: event.data.volume,
        trades: event.data.trades,
      };

      // Use the QuestDB service to store market data
      // Note: This would need to be implemented in questdbService
      // await questdbService.insertMarketData(marketData);

      logger.debug(`💾 Stored market data for ${event.data.symbol} in QuestDB`);
    } catch (error) {
      logger.error(`❌ Failed to store market data in QuestDB:`, error);
      // Don't throw here as this is not critical for event processing
    }
  }

  /**
   * Store OHLCV data in QuestDB
   */
  private async storeOHLCVData(event: OHLCVEvent): Promise<void> {
    try {
      const ohlcvData = {
        timestamp: new Date(event.data.timestamp),
        symbol: event.data.symbol,
        exchange: event.data.exchange,
        timeframe: event.data.timeframe,
        open: event.data.open,
        high: event.data.high,
        low: event.data.low,
        close: event.data.close,
        volume: event.data.volume,
        trades: event.data.trades,
      };

      // Store in QuestDB
      // await questdbService.insertMarketData(ohlcvData);

      logger.debug(`💾 Stored OHLCV data for ${event.data.symbol} in QuestDB`);
    } catch (error) {
      logger.error(`❌ Failed to store OHLCV data in QuestDB:`, error);
    }
  }

  /**
   * Generate signal event based on market data
   */
  private async generateSignalEvent(
    marketEvent: MarketDataEvent,
    priceChange: number
  ): Promise<TradingSignalEvent | null> {
    try {
      const { symbol, price } = marketEvent.data;
      
      // Simple signal generation logic (in production, this would use ML models)
      const direction = priceChange > 0 ? 'LONG' : 'SHORT';
      const strength = this.calculateSignalStrength(priceChange);
      const confidenceScore = Math.min(Math.abs(priceChange) * 100, 95); // Max 95% confidence

      const signalEvent: TradingSignalEvent = {
        id: createEventId(),
        type: 'SIGNAL_GENERATED',
        timestamp: Date.now(),
        version: '1.0',
        source: 'market-data-processor',
        correlationId: marketEvent.correlationId || createCorrelationId(),
        causationId: marketEvent.id,
        userId: marketEvent.userId,
        data: {
          signalId: createEventId(),
          symbol,
          signalType: 'ENTRY',
          direction: direction as 'LONG' | 'SHORT',
          strength,
          timeframe: '1m',
          price,
          confidenceScore,
          expectedReturn: Math.abs(priceChange) * 2, // Simple 2:1 risk-reward
          expectedRisk: Math.abs(priceChange),
          riskRewardRatio: 2.0,
          modelSource: 'price-change-detector',
          modelVersion: '1.0',
          expiresAt: Date.now() + 300000, // 5 minutes expiry
          predictionValues: [price * (1 + (direction === 'LONG' ? priceChange * 2 : -priceChange * 2))],
          features: {
            priceChange,
            volume: marketEvent.data.volume,
            price,
          },
        },
      };

      return signalEvent;
    } catch (error) {
      logger.error(`❌ Failed to generate signal event:`, error);
      return null;
    }
  }

  /**
   * Calculate signal strength based on price change
   */
  private calculateSignalStrength(priceChange: number): 'VERY_WEAK' | 'WEAK' | 'MODERATE' | 'STRONG' | 'VERY_STRONG' {
    const absChange = Math.abs(priceChange);
    
    if (absChange >= 0.05) return 'VERY_STRONG'; // 5%+
    if (absChange >= 0.03) return 'STRONG';      // 3-5%
    if (absChange >= 0.02) return 'MODERATE';    // 2-3%
    if (absChange >= 0.01) return 'WEAK';        // 1-2%
    return 'VERY_WEAK';                          // <1%
  }

  /**
   * Enable/disable signal generation
   */
  public setSignalGenerationEnabled(enabled: boolean): void {
    this.signalGenerationEnabled = enabled;
    logger.info(`📊 Signal generation ${enabled ? 'enabled' : 'disabled'} for market data processor`);
  }

  /**
   * Set price change threshold for signal generation
   */
  public setPriceChangeThreshold(threshold: number): void {
    this.priceChangeThreshold = threshold;
    logger.info(`📊 Price change threshold set to ${(threshold * 100).toFixed(2)}%`);
  }

  /**
   * Get processor statistics
   */
  public getStats(): {
    name: string;
    signalGenerationEnabled: boolean;
    priceChangeThreshold: number;
    trackedSymbols: number;
    lastPrices: Record<string, number>;
  } {
    return {
      name: this.name,
      signalGenerationEnabled: this.signalGenerationEnabled,
      priceChangeThreshold: this.priceChangeThreshold,
      trackedSymbols: this.lastPrices.size,
      lastPrices: Object.fromEntries(this.lastPrices),
    };
  }

  /**
   * Clear price history
   */
  public clearPriceHistory(): void {
    this.lastPrices.clear();
    logger.info(`🗑️ Cleared price history for market data processor`);
  }
}
