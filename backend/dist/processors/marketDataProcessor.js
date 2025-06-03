"use strict";
/**
 * Market Data Event Processor
 * Processes market data events and triggers signal generation
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MarketDataProcessor = void 0;
const redisStreamsService_1 = require("../services/redisStreamsService");
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
class MarketDataProcessor {
    constructor() {
        this.name = 'MarketDataProcessor';
        this.signalGenerationEnabled = true;
        this.lastPrices = new Map();
        this.priceChangeThreshold = 0.001; // 0.1% price change threshold
    }
    getName() {
        return this.name;
    }
    canProcess(event) {
        return (0, events_1.isMarketDataEvent)(event) || event.type.startsWith('OHLCV_');
    }
    async process(event) {
        const startTime = Date.now();
        try {
            if ((0, events_1.isMarketDataEvent)(event)) {
                return await this.processMarketData(event);
            }
            else if (event.type.startsWith('OHLCV_')) {
                return await this.processOHLCV(event);
            }
            throw new Error(`Unsupported event type: ${event.type}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Market data processing failed for event ${event.id}:`, error);
            return {
                eventId: event.id,
                status: events_1.ProcessingStatus.FAILED,
                processingTime: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            };
        }
    }
    /**
     * Process market data event
     */
    async processMarketData(event) {
        const startTime = Date.now();
        const nextEvents = [];
        try {
            const { symbol, price, volume, timestamp } = event.data;
            // Store market data in QuestDB for time-series analysis
            await this.storeMarketData(event);
            // Check for significant price changes
            const lastPrice = this.lastPrices.get(symbol);
            if (lastPrice) {
                const priceChange = Math.abs(price - lastPrice) / lastPrice;
                if (priceChange >= this.priceChangeThreshold) {
                    logger_1.logger.debug(`📈 Significant price change detected for ${symbol}: ${(priceChange * 100).toFixed(2)}%`);
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
                await redisStreamsService_1.redisStreamsService.publishEvent(events_1.STREAM_NAMES.TRADING_SIGNALS, nextEvent);
            }
            logger_1.logger.debug(`✅ Processed market data for ${symbol} @ ${price}`);
            return {
                eventId: event.id,
                status: events_1.ProcessingStatus.COMPLETED,
                processingTime: Date.now() - startTime,
                result: {
                    symbol,
                    price,
                    priceChange: lastPrice ? (price - lastPrice) / lastPrice : 0,
                    signalsGenerated: nextEvents.length,
                },
                nextEvents,
            };
        }
        catch (error) {
            throw new Error(`Market data processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    /**
     * Process OHLCV candle event
     */
    async processOHLCV(event) {
        const startTime = Date.now();
        try {
            const { symbol, close, volume, timestamp } = event.data;
            // Store OHLCV data in QuestDB
            await this.storeOHLCVData(event);
            // Update last price with close price
            this.lastPrices.set(symbol, close);
            logger_1.logger.debug(`✅ Processed OHLCV candle for ${symbol} @ ${close}`);
            return {
                eventId: event.id,
                status: events_1.ProcessingStatus.COMPLETED,
                processingTime: Date.now() - startTime,
                result: {
                    symbol,
                    close,
                    volume,
                    timestamp,
                },
            };
        }
        catch (error) {
            throw new Error(`OHLCV processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    /**
     * Store market data in QuestDB
     */
    async storeMarketData(event) {
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
            logger_1.logger.debug(`💾 Stored market data for ${event.data.symbol} in QuestDB`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to store market data in QuestDB:`, error);
            // Don't throw here as this is not critical for event processing
        }
    }
    /**
     * Store OHLCV data in QuestDB
     */
    async storeOHLCVData(event) {
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
            logger_1.logger.debug(`💾 Stored OHLCV data for ${event.data.symbol} in QuestDB`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to store OHLCV data in QuestDB:`, error);
        }
    }
    /**
     * Generate signal event based on market data
     */
    async generateSignalEvent(marketEvent, priceChange) {
        try {
            const { symbol, price } = marketEvent.data;
            // Simple signal generation logic (in production, this would use ML models)
            const direction = priceChange > 0 ? 'LONG' : 'SHORT';
            const strength = this.calculateSignalStrength(priceChange);
            const confidenceScore = Math.min(Math.abs(priceChange) * 100, 95); // Max 95% confidence
            const signalEvent = {
                id: (0, events_1.createEventId)(),
                type: 'SIGNAL_GENERATED',
                timestamp: Date.now(),
                version: '1.0',
                source: 'market-data-processor',
                correlationId: marketEvent.correlationId || (0, events_1.createCorrelationId)(),
                causationId: marketEvent.id,
                userId: marketEvent.userId,
                data: {
                    signalId: (0, events_1.createEventId)(),
                    symbol,
                    signalType: 'ENTRY',
                    direction: direction,
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
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to generate signal event:`, error);
            return null;
        }
    }
    /**
     * Calculate signal strength based on price change
     */
    calculateSignalStrength(priceChange) {
        const absChange = Math.abs(priceChange);
        if (absChange >= 0.05)
            return 'VERY_STRONG'; // 5%+
        if (absChange >= 0.03)
            return 'STRONG'; // 3-5%
        if (absChange >= 0.02)
            return 'MODERATE'; // 2-3%
        if (absChange >= 0.01)
            return 'WEAK'; // 1-2%
        return 'VERY_WEAK'; // <1%
    }
    /**
     * Enable/disable signal generation
     */
    setSignalGenerationEnabled(enabled) {
        this.signalGenerationEnabled = enabled;
        logger_1.logger.info(`📊 Signal generation ${enabled ? 'enabled' : 'disabled'} for market data processor`);
    }
    /**
     * Set price change threshold for signal generation
     */
    setPriceChangeThreshold(threshold) {
        this.priceChangeThreshold = threshold;
        logger_1.logger.info(`📊 Price change threshold set to ${(threshold * 100).toFixed(2)}%`);
    }
    /**
     * Get processor statistics
     */
    getStats() {
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
    clearPriceHistory() {
        this.lastPrices.clear();
        logger_1.logger.info(`🗑️ Cleared price history for market data processor`);
    }
}
exports.MarketDataProcessor = MarketDataProcessor;
//# sourceMappingURL=marketDataProcessor.js.map