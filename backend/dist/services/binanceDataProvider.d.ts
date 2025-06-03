/**
 * Binance Market Data Provider
 * Fetches real historical data from Binance public API
 */
import { MarketDataProvider, MarketDataRequest, MarketDataResponse } from '../types/marketData';
export declare class BinanceDataProvider implements MarketDataProvider {
    readonly name = "binance";
    private readonly baseUrl;
    private readonly rateLimitDelay;
    /**
     * Check if Binance API is available
     */
    isAvailable(): boolean;
    /**
     * Fetch historical data from Binance
     */
    fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse>;
    /**
     * Fetch kline data from Binance API
     */
    private fetchBinanceKlines;
    /**
     * Convert Binance data to our MarketDataPoint format
     */
    private convertBinanceDataToMarketData;
    /**
     * Convert our symbol format to Binance format
     */
    private convertSymbolToBinanceFormat;
    /**
     * Convert our timeframe to Binance interval format
     */
    private convertTimeframeToBinanceInterval;
    /**
     * Generate fallback mock data if real data fails
     */
    private generateFallbackData;
    /**
     * Get interval in milliseconds for a timeframe
     */
    private getIntervalMs;
    /**
     * Delay utility for rate limiting
     */
    private delay;
}
export declare function createBinanceDataProvider(): BinanceDataProvider;
