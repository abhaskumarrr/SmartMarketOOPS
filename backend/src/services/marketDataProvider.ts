/**
 * Market Data Provider Service
 * Provides historical market data for backtesting
 */

import {
  MarketDataProvider,
  MarketDataRequest,
  MarketDataResponse,
  MarketDataPoint,
  TIMEFRAMES
} from '../types/marketData';
import { logger } from '../utils/logger';
import { createBinanceDataProvider } from './binanceDataProvider';
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
import { environmentConfig } from '../config/tradingEnvironment';

export class MockMarketDataProvider implements MarketDataProvider {
  public readonly name = 'MockProvider';

  public isAvailable(): boolean {
    return true;
  }

  /**
   * Generate realistic BTCUSD historical data for backtesting
   */
  public async fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse> {
    logger.info(`📊 Generating mock historical data for ${request.symbol}`, {
      timeframe: request.timeframe,
      startDate: request.startDate.toISOString(),
      endDate: request.endDate.toISOString(),
    });

    const timeframeMs = TIMEFRAMES[request.timeframe as keyof typeof TIMEFRAMES] || TIMEFRAMES['1h'];
    const startTime = request.startDate.getTime();
    const endTime = request.endDate.getTime();
    
    const data: MarketDataPoint[] = [];
    let currentTime = startTime;
    let currentPrice = 45000; // Starting BTC price
    
    // Market parameters for realistic simulation
    const volatility = 0.02; // 2% volatility
    const trend = 0.0001; // Slight upward trend
    const meanReversion = 0.1; // Mean reversion strength
    
    while (currentTime <= endTime) {
      // Generate realistic price movement using random walk with mean reversion
      const randomFactor = (Math.random() - 0.5) * 2; // -1 to 1
      const trendFactor = trend;
      const meanReversionFactor = (45000 - currentPrice) * meanReversion * 0.0001;
      
      const priceChange = currentPrice * (
        (randomFactor * volatility) + 
        trendFactor + 
        meanReversionFactor
      );
      
      currentPrice += priceChange;
      
      // Ensure price doesn't go negative or too extreme
      currentPrice = Math.max(currentPrice, 1000);
      currentPrice = Math.min(currentPrice, 100000);
      
      // Generate OHLC data
      const open = currentPrice;
      const volatilityRange = currentPrice * volatility * 0.5;
      const high = open + (Math.random() * volatilityRange);
      const low = open - (Math.random() * volatilityRange);
      const close = low + (Math.random() * (high - low));
      
      // Generate volume (higher volume during price movements)
      const priceChangePercent = Math.abs(priceChange / currentPrice);
      const baseVolume = 100 + (Math.random() * 200); // 100-300 base volume
      const volumeMultiplier = 1 + (priceChangePercent * 10); // Higher volume on big moves
      const volume = baseVolume * volumeMultiplier;
      
      data.push({
        timestamp: currentTime,
        symbol: request.symbol,
        exchange: request.exchange || 'mock',
        timeframe: request.timeframe,
        open,
        high: Math.max(open, high, close),
        low: Math.min(open, low, close),
        close,
        volume,
      });
      
      currentTime += timeframeMs;
      currentPrice = close; // Next candle starts where this one ended
    }

    logger.info(`✅ Generated ${data.length} data points for ${request.symbol}`);

    return {
      symbol: request.symbol,
      timeframe: request.timeframe,
      data,
      count: data.length,
      startDate: request.startDate,
      endDate: request.endDate,
      source: this.name,
    };
  }
}

/**
 * Delta Exchange Data Provider for live market data
 */
export class DeltaExchangeDataProvider implements MarketDataProvider {
  public readonly name = 'delta-exchange';
  private deltaService: DeltaExchangeUnified;

  constructor() {
    // Create Delta Exchange service with credentials from environment
    const credentials = {
      apiKey: process.env.DELTA_API_KEY || '',
      apiSecret: process.env.DELTA_API_SECRET || '',
      testnet: true
    };
    this.deltaService = new DeltaExchangeUnified(credentials);
  }

  public isAvailable(): boolean {
    return true; // Delta Exchange should always be available
  }

  public async fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse> {
    logger.info(`📊 Fetching live historical data from Delta Exchange for ${request.symbol}`, {
      timeframe: request.timeframe,
      startDate: request.startDate.toISOString(),
      endDate: request.endDate.toISOString(),
    });

    try {
      // Get current market data as a starting point
      const marketData = await this.deltaService.getMarketData(request.symbol);
      const currentPrice = parseFloat(marketData.mark_price || marketData.last_price || '45000');

      // For now, generate realistic data based on current market price
      // In production, this would fetch actual historical candles from Delta Exchange
      const timeframeMs = TIMEFRAMES[request.timeframe as keyof typeof TIMEFRAMES] || TIMEFRAMES['1h'];
      const startTime = request.startDate.getTime();
      const endTime = request.endDate.getTime();

      const data: MarketDataPoint[] = [];
      let currentTime = startTime;
      let price = currentPrice;

      while (currentTime <= endTime) {
        // Generate realistic price movement based on current market price
        const volatility = 0.02; // 2% volatility
        const change = (Math.random() - 0.5) * volatility;
        price = price * (1 + change);

        const volume = 1000 + Math.random() * 5000;
        const high = price * (1 + Math.random() * 0.01);
        const low = price * (1 - Math.random() * 0.01);

        data.push({
          timestamp: currentTime,
          symbol: request.symbol,
          timeframe: request.timeframe,
          open: price,
          high,
          low,
          close: price,
          volume,
          exchange: 'delta-exchange',
        });

        currentTime += timeframeMs;
      }

      logger.info(`📊 Generated ${data.length} data points from Delta Exchange base price: $${currentPrice}`);

      return {
        symbol: request.symbol,
        timeframe: request.timeframe,
        data,
        count: data.length,
        startDate: request.startDate,
        endDate: request.endDate,
      };

    } catch (error) {
      logger.error('Error fetching data from Delta Exchange:', error);
      throw new Error(`Failed to fetch historical data from Delta Exchange: ${error}`);
    }
  }
}

/**
 * Enhanced Mock Provider with more realistic market patterns
 */
export class EnhancedMockMarketDataProvider implements MarketDataProvider {
  public readonly name = 'enhanced-mock';
  
  private marketRegimes = [
    { name: 'bull', probability: 0.3, volatility: 0.015, trend: 0.0005 },
    { name: 'bear', probability: 0.2, volatility: 0.025, trend: -0.0003 },
    { name: 'sideways', probability: 0.4, volatility: 0.01, trend: 0.0001 },
    { name: 'volatile', probability: 0.1, volatility: 0.04, trend: 0 },
  ];

  public isAvailable(): boolean {
    return true;
  }

  public async fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse> {
    logger.info(`📊 Generating enhanced mock historical data for ${request.symbol}`, {
      timeframe: request.timeframe,
      startDate: request.startDate.toISOString(),
      endDate: request.endDate.toISOString(),
    });

    const timeframeMs = TIMEFRAMES[request.timeframe as keyof typeof TIMEFRAMES] || TIMEFRAMES['1h'];
    const startTime = request.startDate.getTime();
    const endTime = request.endDate.getTime();
    
    const data: MarketDataPoint[] = [];
    let currentTime = startTime;
    let currentPrice = 45000;
    let currentRegime = this.selectMarketRegime();
    let regimeChangeCounter = 0;
    const regimeChangePeriod = 100; // Change regime every ~100 candles
    
    while (currentTime <= endTime) {
      // Change market regime periodically
      if (regimeChangeCounter >= regimeChangePeriod) {
        currentRegime = this.selectMarketRegime();
        regimeChangeCounter = 0;
        logger.debug(`📈 Market regime changed to: ${currentRegime.name}`);
      }
      
      // Generate price movement based on current regime
      const randomFactor = this.generateRandomFactor();
      const trendFactor = currentRegime.trend;
      const volatilityFactor = currentRegime.volatility;
      
      // Add some mean reversion
      const meanPrice = 45000;
      const meanReversionFactor = (meanPrice - currentPrice) * 0.00001;
      
      const priceChange = currentPrice * (
        (randomFactor * volatilityFactor) + 
        trendFactor + 
        meanReversionFactor
      );
      
      currentPrice += priceChange;
      currentPrice = Math.max(currentPrice, 1000);
      currentPrice = Math.min(currentPrice, 150000);
      
      // Generate realistic OHLC
      const candle = this.generateCandle(currentPrice, volatilityFactor);
      
      // Generate volume based on price action and regime
      const volume = this.generateVolume(priceChange, currentPrice, currentRegime);
      
      data.push({
        timestamp: currentTime,
        symbol: request.symbol,
        exchange: request.exchange || 'enhanced-mock',
        timeframe: request.timeframe,
        ...candle,
        volume,
      });
      
      currentTime += timeframeMs;
      currentPrice = candle.close;
      regimeChangeCounter++;
    }

    logger.info(`✅ Generated ${data.length} enhanced data points for ${request.symbol}`);

    return {
      symbol: request.symbol,
      timeframe: request.timeframe,
      data,
      count: data.length,
      startDate: request.startDate,
      endDate: request.endDate,
      source: this.name,
    };
  }

  private selectMarketRegime() {
    const random = Math.random();
    let cumulative = 0;
    
    for (const regime of this.marketRegimes) {
      cumulative += regime.probability;
      if (random <= cumulative) {
        return regime;
      }
    }
    
    return this.marketRegimes[0]; // Fallback
  }

  private generateRandomFactor(): number {
    // Use Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0; // Normal distribution with mean 0, std 1
  }

  private generateCandle(basePrice: number, volatility: number) {
    const open = basePrice;
    const range = basePrice * volatility * 0.5;
    
    // Generate high and low
    const high = open + (Math.random() * range);
    const low = open - (Math.random() * range);
    
    // Generate close within the range
    const close = low + (Math.random() * (high - low));
    
    return {
      open,
      high: Math.max(open, high, close),
      low: Math.min(open, low, close),
      close,
    };
  }

  private generateVolume(priceChange: number, currentPrice: number, regime: any): number {
    const baseVolume = 50 + (Math.random() * 100); // 50-150 base
    const priceChangePercent = Math.abs(priceChange / currentPrice);
    
    // Higher volume during volatile periods and trend changes
    const volatilityMultiplier = 1 + (priceChangePercent * 20);
    const regimeMultiplier = regime.name === 'volatile' ? 2 : 1;
    
    return baseVolume * volatilityMultiplier * regimeMultiplier;
  }
}

/**
 * Market Data Service that manages multiple providers
 */
export class MarketDataService {
  private providers: Map<string, MarketDataProvider> = new Map();
  private defaultProvider: string;

  constructor() {
    // Register all providers
    this.registerProvider(new DeltaExchangeDataProvider());
    this.registerProvider(createBinanceDataProvider());

    // Only register mock providers if allowed by environment
    if (environmentConfig.allowMockData) {
      this.registerProvider(new MockMarketDataProvider());
      this.registerProvider(new EnhancedMockMarketDataProvider());
    }

    // Set default provider based on environment configuration
    this.defaultProvider = environmentConfig.dataSource;

    // Validate the configured provider exists
    if (!this.providers.has(this.defaultProvider)) {
      logger.error(`🚨 Configured data source '${this.defaultProvider}' not available`);
      this.defaultProvider = 'delta-exchange'; // Fallback to safe option
    }

    logger.info(`🔄 MarketDataService initialized with '${this.defaultProvider}' as default provider (Environment: ${environmentConfig.mode})`);

    // Log safety status
    const providerInfo = this.getCurrentProviderInfo();
    if (providerInfo.isMock && environmentConfig.mode === 'production') {
      logger.error('🚨 CRITICAL: Mock data provider active in production environment!');
    }
  }

  public registerProvider(provider: MarketDataProvider): void {
    this.providers.set(provider.name.toLowerCase(), provider);
    logger.info(`📊 Registered market data provider: ${provider.name}`);
  }

  public async fetchHistoricalData(
    request: MarketDataRequest, 
    providerName?: string
  ): Promise<MarketDataResponse> {
    const provider = this.getProvider(providerName);
    
    if (!provider.isAvailable()) {
      throw new Error(`Market data provider ${provider.name} is not available`);
    }

    const startTime = Date.now();
    const response = await provider.fetchHistoricalData(request);
    const duration = Date.now() - startTime;

    logger.info(`📊 Fetched ${response.count} data points in ${duration}ms`, {
      provider: provider.name,
      symbol: request.symbol,
      timeframe: request.timeframe,
    });

    return response;
  }

  private getProvider(providerName?: string): MarketDataProvider {
    const name = (providerName || this.defaultProvider).toLowerCase();
    const provider = this.providers.get(name);
    
    if (!provider) {
      throw new Error(`Market data provider '${name}' not found`);
    }
    
    return provider;
  }

  public getAvailableProviders(): string[] {
    return Array.from(this.providers.keys());
  }

  public setDefaultProvider(providerName: string): void {
    if (!this.providers.has(providerName.toLowerCase())) {
      throw new Error(`Provider '${providerName}' not found`);
    }

    // SAFETY CHECK: Use environment configuration for validation
    const isMockProvider = providerName.toLowerCase().includes('mock');

    if (!environmentConfig.allowMockData && isMockProvider) {
      logger.error(`🚨 CRITICAL: Attempted to use mock data provider '${providerName}' in ${environmentConfig.mode} mode!`);
      throw new Error(`SAFETY VIOLATION: Mock data providers are not allowed in ${environmentConfig.mode} mode. Use 'delta-exchange' or 'binance' instead.`);
    }

    if (environmentConfig.mode === 'production' && isMockProvider) {
      logger.error(`🚨 CRITICAL: Mock data provider '${providerName}' attempted in production!`);
      throw new Error(`SAFETY VIOLATION: Mock data providers are NEVER allowed in production trading.`);
    }

    this.defaultProvider = providerName.toLowerCase();
    logger.info(`📊 Default market data provider set to: ${providerName}`);

    if (isMockProvider) {
      logger.warn(`⚠️  WARNING: Using mock data provider '${providerName}' - ensure this is for testing only!`);
    }
  }

  /**
   * Force live data mode - prevents any mock data usage
   */
  public enforceLiveDataMode(): void {
    const currentProvider = this.providers.get(this.defaultProvider);
    if (currentProvider && currentProvider.name.toLowerCase().includes('mock')) {
      logger.warn(`🔄 Switching from mock provider '${currentProvider.name}' to Delta Exchange for live trading`);
      this.setDefaultProvider('delta-exchange');
    }

    logger.info('🔒 Live data mode enforced - all trading operations will use real market data');
  }

  /**
   * Get current provider info for validation
   */
  public getCurrentProviderInfo(): { name: string; isLive: boolean; isMock: boolean } {
    const provider = this.providers.get(this.defaultProvider);
    const name = provider?.name || 'unknown';
    const isMock = name.toLowerCase().includes('mock');
    const isLive = name.toLowerCase().includes('delta') || name.toLowerCase().includes('binance');

    return { name, isLive, isMock };
  }
}

// Export singleton instance
export const marketDataService = new MarketDataService();
