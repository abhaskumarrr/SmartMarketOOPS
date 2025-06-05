/**
 * Market Data Provider Service
 * Provides historical market data for backtesting
 */
import { MarketDataProvider, MarketDataRequest, MarketDataResponse } from '../types/marketData';
export declare class MockMarketDataProvider implements MarketDataProvider {
    readonly name = "MockProvider";
    isAvailable(): boolean;
    /**
     * Generate realistic BTCUSD historical data for backtesting
     */
    fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse>;
}
/**
 * Delta Exchange Data Provider for live market data
 */
export declare class DeltaExchangeDataProvider implements MarketDataProvider {
    readonly name = "delta-exchange";
    private deltaService;
    constructor();
    isAvailable(): boolean;
    fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse>;
}
/**
 * Enhanced Mock Provider with more realistic market patterns
 */
export declare class EnhancedMockMarketDataProvider implements MarketDataProvider {
    readonly name = "enhanced-mock";
    private marketRegimes;
    isAvailable(): boolean;
    fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse>;
    private selectMarketRegime;
    private generateRandomFactor;
    private generateCandle;
    private generateVolume;
}
/**
 * Market Data Service that manages multiple providers
 */
export declare class MarketDataService {
    private providers;
    private defaultProvider;
    constructor();
    registerProvider(provider: MarketDataProvider): void;
    fetchHistoricalData(request: MarketDataRequest, providerName?: string): Promise<MarketDataResponse>;
    private getProvider;
    getAvailableProviders(): string[];
    setDefaultProvider(providerName: string): void;
    /**
     * Force live data mode - prevents any mock data usage
     */
    enforceLiveDataMode(): void;
    /**
     * Get current provider info for validation
     */
    getCurrentProviderInfo(): {
        name: string;
        isLive: boolean;
        isMock: boolean;
    };
}
export declare const marketDataService: MarketDataService;
