/**
 * Market Data Event Processor
 * Processes market data events and triggers signal generation
 */
import { EventProcessor } from '../services/eventProcessingPipeline';
import { TradingEvent, EventProcessingResult } from '../types/events';
export declare class MarketDataProcessor implements EventProcessor {
    private name;
    private signalGenerationEnabled;
    private lastPrices;
    private priceChangeThreshold;
    getName(): string;
    canProcess(event: TradingEvent): boolean;
    process(event: TradingEvent): Promise<EventProcessingResult>;
    /**
     * Process market data event
     */
    private processMarketData;
    /**
     * Process OHLCV candle event
     */
    private processOHLCV;
    /**
     * Store market data in QuestDB
     */
    private storeMarketData;
    /**
     * Store OHLCV data in QuestDB
     */
    private storeOHLCVData;
    /**
     * Generate signal event based on market data
     */
    private generateSignalEvent;
    /**
     * Calculate signal strength based on price change
     */
    private calculateSignalStrength;
    /**
     * Enable/disable signal generation
     */
    setSignalGenerationEnabled(enabled: boolean): void;
    /**
     * Set price change threshold for signal generation
     */
    setPriceChangeThreshold(threshold: number): void;
    /**
     * Get processor statistics
     */
    getStats(): {
        name: string;
        signalGenerationEnabled: boolean;
        priceChangeThreshold: number;
        trackedSymbols: number;
        lastPrices: Record<string, number>;
    };
    /**
     * Clear price history
     */
    clearPriceHistory(): void;
}
