/**
 * Multi-Timeframe Multi-Asset Data Provider
 * Combines multi-timeframe analysis with multi-asset support for comprehensive market data
 */

import { MarketDataPoint } from '../types/marketData';
import { createMultiAssetDataProvider, CryptoPair, AssetConfig } from './multiAssetDataProvider';
import { MultiTimeframeDataProvider, Timeframe, MultiTimeframeData } from './multiTimeframeDataProvider';
import { logger } from '../utils/logger';

// Forward declaration - will be imported from backtester
export interface TimeframeAssetConfig {
  asset: CryptoPair;
  timeframes: Timeframe[];
  priority: 'PRIMARY' | 'SECONDARY' | 'CONFIRMATION';
  weight: number;
}

export interface MultiTimeframeMultiAssetData {
  timestamp: number;
  assets: {
    [asset in CryptoPair]?: {
      [timeframe in Timeframe]?: MarketDataPoint;
    };
  };
  crossAssetAnalysis: {
    correlations: {
      [timeframePair: string]: {
        btc_eth: number;
        btc_sol: number;
        eth_sol: number;
      };
    };
    dominance: {
      [timeframe in Timeframe]?: {
        btc: number;
        eth: number;
        sol: number;
      };
    };
    volatilityRanking: {
      [timeframe in Timeframe]?: CryptoPair[];
    };
  };
  timeframeConsensus: {
    [asset in CryptoPair]?: {
      bullishTimeframes: Timeframe[];
      bearishTimeframes: Timeframe[];
      neutralTimeframes: Timeframe[];
      overallSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
      consensusStrength: number;
    };
  };
}



export class MultiTimeframeMultiAssetDataProvider {
  private multiAssetProvider = createMultiAssetDataProvider();
  private multiTimeframeProvider = new MultiTimeframeDataProvider();
  private supportedTimeframes: Timeframe[] = ['1m', '3m', '5m', '15m', '1h', '4h', '1d'];
  private supportedAssets: CryptoPair[] = ['BTCUSD', 'ETHUSD', 'SOLUSD'];

  constructor() {
    logger.info('🔄 Multi-Timeframe Multi-Asset Data Provider initialized', {
      supportedAssets: this.supportedAssets,
      supportedTimeframes: this.supportedTimeframes,
    });
  }

  /**
   * Fetch comprehensive multi-timeframe multi-asset data
   */
  public async fetchComprehensiveData(
    startDate: Date,
    endDate: Date,
    assetConfigs: TimeframeAssetConfig[],
    primaryTimeframe: Timeframe = '1h'
  ): Promise<MultiTimeframeMultiAssetData[]> {
    
    logger.info('📊 Fetching comprehensive multi-timeframe multi-asset data...', {
      period: `${startDate.toISOString().split('T')[0]} to ${endDate.toISOString().split('T')[0]}`,
      assetConfigs: assetConfigs.map(c => `${c.asset}:${c.timeframes.join(',')}`),
      primaryTimeframe,
    });

    // Step 1: Fetch base data for all assets and timeframes
    const baseData = await this.fetchBaseData(startDate, endDate, assetConfigs);

    // Step 2: Align data across timeframes and assets
    const alignedData = this.alignMultiTimeframeMultiAssetData(baseData, primaryTimeframe);

    // Step 3: Generate cross-asset analysis
    const enhancedData = this.enhanceWithCrossAssetAnalysis(alignedData);

    // Step 4: Add timeframe consensus analysis
    const finalData = this.addTimeframeConsensus(enhancedData);

    logger.info('✅ Comprehensive data fetching completed', {
      dataPoints: finalData.length,
      assetsProcessed: assetConfigs.length,
      timeframesProcessed: this.getUniqueTimeframes(assetConfigs).length,
    });

    return finalData;
  }

  /**
   * Fetch base data for all asset-timeframe combinations
   */
  private async fetchBaseData(
    startDate: Date,
    endDate: Date,
    assetConfigs: TimeframeAssetConfig[]
  ): Promise<{ [key: string]: MarketDataPoint[] }> {
    
    const baseData: { [key: string]: MarketDataPoint[] } = {};
    const fetchPromises: Promise<void>[] = [];

    // Create unique asset-timeframe combinations
    const combinations = this.createAssetTimeframeCombinations(assetConfigs);

    for (const combination of combinations) {
      const { asset, timeframe } = combination;
      const key = `${asset}_${timeframe}`;

      const promise = this.fetchSingleAssetTimeframeData(asset, timeframe, startDate, endDate)
        .then(data => {
          baseData[key] = data;
          logger.debug(`✅ Fetched ${data.length} candles for ${asset} ${timeframe}`);
        })
        .catch(error => {
          logger.error(`❌ Failed to fetch data for ${asset} ${timeframe}:`, error);
          baseData[key] = [];
        });

      fetchPromises.push(promise);
    }

    await Promise.all(fetchPromises);

    const totalCandles = Object.values(baseData).reduce((sum, data) => sum + data.length, 0);
    logger.info('📊 Base data fetching completed', {
      combinations: combinations.length,
      totalCandles,
      successfulFetches: Object.values(baseData).filter(data => data.length > 0).length,
    });

    return baseData;
  }

  /**
   * Fetch data for a single asset-timeframe combination
   */
  private async fetchSingleAssetTimeframeData(
    asset: CryptoPair,
    timeframe: Timeframe,
    startDate: Date,
    endDate: Date
  ): Promise<MarketDataPoint[]> {
    
    try {
      // Use multi-asset provider for data fetching
      const assetData = await this.multiAssetProvider.fetchMultiAssetData(
        timeframe,
        startDate,
        endDate,
        [asset]
      );

      return assetData[asset] || [];
    } catch (error) {
      logger.warn(`⚠️ Failed to fetch real data for ${asset} ${timeframe}, using fallback`);
      return this.generateFallbackData(asset, timeframe, startDate, endDate);
    }
  }

  /**
   * Generate fallback data when real data is unavailable
   */
  private generateFallbackData(
    asset: CryptoPair,
    timeframe: Timeframe,
    startDate: Date,
    endDate: Date
  ): MarketDataPoint[] {
    
    const timeframeMinutes = this.getTimeframeMinutes(timeframe);
    const totalMinutes = (endDate.getTime() - startDate.getTime()) / (1000 * 60);
    const candleCount = Math.floor(totalMinutes / timeframeMinutes);

    const basePrice = this.getAssetBasePrice(asset);
    const data: MarketDataPoint[] = [];

    for (let i = 0; i < candleCount; i++) {
      const timestamp = startDate.getTime() + (i * timeframeMinutes * 60 * 1000);
      const volatility = this.getAssetVolatility(asset);
      
      const priceChange = (Math.random() - 0.5) * volatility * basePrice;
      const currentPrice = basePrice + priceChange;

      data.push({
        symbol: asset,
        exchange: 'binance',
        timeframe: timeframe,
        timestamp,
        open: currentPrice * (0.999 + Math.random() * 0.002),
        high: currentPrice * (1.001 + Math.random() * 0.004),
        low: currentPrice * (0.999 - Math.random() * 0.004),
        close: currentPrice,
        volume: 1000000 + Math.random() * 5000000,
      });
    }

    return data;
  }

  /**
   * Align data across multiple timeframes and assets
   */
  private alignMultiTimeframeMultiAssetData(
    baseData: { [key: string]: MarketDataPoint[] },
    primaryTimeframe: Timeframe
  ): MultiTimeframeMultiAssetData[] {
    
    logger.info('🔄 Aligning multi-timeframe multi-asset data...', { primaryTimeframe });

    // Find the primary timeframe data to use as the base timeline
    const primaryKeys = Object.keys(baseData).filter(key => key.endsWith(`_${primaryTimeframe}`));
    
    if (primaryKeys.length === 0) {
      throw new Error(`No data found for primary timeframe: ${primaryTimeframe}`);
    }

    // Use the first primary asset as the timeline reference
    const referenceKey = primaryKeys[0];
    const referenceData = baseData[referenceKey];
    const alignedData: MultiTimeframeMultiAssetData[] = [];

    for (let i = 0; i < referenceData.length; i++) {
      const referenceCandle = referenceData[i];
      const timestamp = referenceCandle.timestamp;

      const dataPoint: MultiTimeframeMultiAssetData = {
        timestamp,
        assets: {},
        crossAssetAnalysis: {
          correlations: {},
          dominance: {},
          volatilityRanking: {},
        },
        timeframeConsensus: {},
      };

      // Populate data for each asset and timeframe
      for (const asset of this.supportedAssets) {
        dataPoint.assets[asset] = {};
        
        for (const timeframe of this.supportedTimeframes) {
          const key = `${asset}_${timeframe}`;
          const assetData = baseData[key];
          
          if (assetData && assetData.length > 0) {
            // Find the closest candle for this timestamp
            const closestCandle = this.findClosestCandle(assetData, timestamp, timeframe);
            if (closestCandle) {
              dataPoint.assets[asset]![timeframe] = closestCandle;
            }
          }
        }
      }

      alignedData.push(dataPoint);
    }

    logger.info('✅ Data alignment completed', {
      alignedDataPoints: alignedData.length,
      referenceTimeframe: primaryTimeframe,
    });

    return alignedData;
  }

  /**
   * Find the closest candle for a given timestamp and timeframe
   */
  private findClosestCandle(
    data: MarketDataPoint[],
    targetTimestamp: number,
    timeframe: Timeframe
  ): MarketDataPoint | null {
    
    const timeframeMinutes = this.getTimeframeMinutes(timeframe);
    const tolerance = timeframeMinutes * 60 * 1000; // Tolerance in milliseconds

    let closestCandle: MarketDataPoint | null = null;
    let minDifference = Infinity;

    for (const candle of data) {
      const difference = Math.abs(candle.timestamp - targetTimestamp);
      
      if (difference <= tolerance && difference < minDifference) {
        minDifference = difference;
        closestCandle = candle;
      }
    }

    return closestCandle;
  }

  /**
   * Enhance data with cross-asset analysis
   */
  private enhanceWithCrossAssetAnalysis(
    data: MultiTimeframeMultiAssetData[]
  ): MultiTimeframeMultiAssetData[] {
    
    logger.info('🔍 Enhancing with cross-asset analysis...');

    for (let i = 0; i < data.length; i++) {
      const dataPoint = data[i];

      // Calculate correlations for each timeframe
      for (const timeframe of this.supportedTimeframes) {
        const correlations = this.calculateTimeframeCorrelations(data, i, timeframe, 20);
        dataPoint.crossAssetAnalysis.correlations[timeframe] = correlations;
      }

      // Calculate market dominance for each timeframe
      for (const timeframe of this.supportedTimeframes) {
        const dominance = this.calculateMarketDominance(dataPoint, timeframe);
        dataPoint.crossAssetAnalysis.dominance[timeframe] = dominance;
      }

      // Calculate volatility ranking for each timeframe
      for (const timeframe of this.supportedTimeframes) {
        const ranking = this.calculateVolatilityRanking(data, i, timeframe, 10);
        dataPoint.crossAssetAnalysis.volatilityRanking[timeframe] = ranking;
      }
    }

    logger.info('✅ Cross-asset analysis enhancement completed');
    return data;
  }

  /**
   * Add timeframe consensus analysis
   */
  private addTimeframeConsensus(
    data: MultiTimeframeMultiAssetData[]
  ): MultiTimeframeMultiAssetData[] {
    
    logger.info('🎯 Adding timeframe consensus analysis...');

    for (let i = 0; i < data.length; i++) {
      const dataPoint = data[i];

      for (const asset of this.supportedAssets) {
        const consensus = this.calculateTimeframeConsensus(dataPoint, asset);
        dataPoint.timeframeConsensus[asset] = consensus;
      }
    }

    logger.info('✅ Timeframe consensus analysis completed');
    return data;
  }

  /**
   * Calculate correlations for a specific timeframe
   */
  private calculateTimeframeCorrelations(
    data: MultiTimeframeMultiAssetData[],
    currentIndex: number,
    timeframe: Timeframe,
    lookback: number
  ): { btc_eth: number; btc_sol: number; eth_sol: number } {
    
    if (currentIndex < lookback) {
      return { btc_eth: 0, btc_sol: 0, eth_sol: 0 };
    }

    const returns: { [asset: string]: number[] } = { btc: [], eth: [], sol: [] };

    // Calculate returns for the lookback period
    for (let i = currentIndex - lookback + 1; i <= currentIndex; i++) {
      const current = data[i];
      const previous = data[i - 1];

      if (current && previous) {
        for (const asset of this.supportedAssets) {
          const assetKey = asset.substring(0, 3).toLowerCase();
          const currentPrice = current.assets[asset]?.[timeframe]?.close;
          const previousPrice = previous.assets[asset]?.[timeframe]?.close;

          if (currentPrice && previousPrice) {
            const returnValue = Math.log(currentPrice / previousPrice);
            returns[assetKey].push(returnValue);
          }
        }
      }
    }

    // Calculate correlations
    return {
      btc_eth: this.calculateCorrelation(returns.btc, returns.eth),
      btc_sol: this.calculateCorrelation(returns.btc, returns.sol),
      eth_sol: this.calculateCorrelation(returns.eth, returns.sol),
    };
  }

  /**
   * Calculate market dominance for a timeframe
   */
  private calculateMarketDominance(
    dataPoint: MultiTimeframeMultiAssetData,
    timeframe: Timeframe
  ): { btc: number; eth: number; sol: number } {
    
    const volumes: { [key: string]: number } = {};
    let totalVolume = 0;

    for (const asset of this.supportedAssets) {
      const assetKey = asset.substring(0, 3).toLowerCase();
      const volume = dataPoint.assets[asset]?.[timeframe]?.volume || 0;
      volumes[assetKey] = volume;
      totalVolume += volume;
    }

    if (totalVolume === 0) {
      return { btc: 0.33, eth: 0.33, sol: 0.33 };
    }

    return {
      btc: volumes.btc / totalVolume,
      eth: volumes.eth / totalVolume,
      sol: volumes.sol / totalVolume,
    };
  }

  /**
   * Calculate volatility ranking for a timeframe
   */
  private calculateVolatilityRanking(
    data: MultiTimeframeMultiAssetData[],
    currentIndex: number,
    timeframe: Timeframe,
    lookback: number
  ): CryptoPair[] {
    
    if (currentIndex < lookback) {
      return [...this.supportedAssets];
    }

    const volatilities: { asset: CryptoPair; volatility: number }[] = [];

    for (const asset of this.supportedAssets) {
      const returns: number[] = [];

      for (let i = currentIndex - lookback + 1; i <= currentIndex; i++) {
        const current = data[i];
        const previous = data[i - 1];

        if (current && previous) {
          const currentPrice = current.assets[asset]?.[timeframe]?.close;
          const previousPrice = previous.assets[asset]?.[timeframe]?.close;

          if (currentPrice && previousPrice) {
            returns.push(Math.log(currentPrice / previousPrice));
          }
        }
      }

      const volatility = this.calculateStandardDeviation(returns);
      volatilities.push({ asset, volatility });
    }

    // Sort by volatility (highest first)
    volatilities.sort((a, b) => b.volatility - a.volatility);
    return volatilities.map(v => v.asset);
  }

  /**
   * Calculate timeframe consensus for an asset
   */
  private calculateTimeframeConsensus(
    dataPoint: MultiTimeframeMultiAssetData,
    asset: CryptoPair
  ): {
    bullishTimeframes: Timeframe[];
    bearishTimeframes: Timeframe[];
    neutralTimeframes: Timeframe[];
    overallSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    consensusStrength: number;
  } {
    
    const bullishTimeframes: Timeframe[] = [];
    const bearishTimeframes: Timeframe[] = [];
    const neutralTimeframes: Timeframe[] = [];

    for (const timeframe of this.supportedTimeframes) {
      const candle = dataPoint.assets[asset]?.[timeframe];
      
      if (candle) {
        const sentiment = this.determineCandleSentiment(candle);
        
        if (sentiment === 'BULLISH') {
          bullishTimeframes.push(timeframe);
        } else if (sentiment === 'BEARISH') {
          bearishTimeframes.push(timeframe);
        } else {
          neutralTimeframes.push(timeframe);
        }
      }
    }

    // Determine overall sentiment
    let overallSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
    
    if (bullishTimeframes.length > bearishTimeframes.length) {
      overallSentiment = 'BULLISH';
    } else if (bearishTimeframes.length > bullishTimeframes.length) {
      overallSentiment = 'BEARISH';
    }

    // Calculate consensus strength
    const totalTimeframes = this.supportedTimeframes.length;
    const majorityCount = Math.max(bullishTimeframes.length, bearishTimeframes.length);
    const consensusStrength = majorityCount / totalTimeframes;

    return {
      bullishTimeframes,
      bearishTimeframes,
      neutralTimeframes,
      overallSentiment,
      consensusStrength,
    };
  }

  // Helper methods
  private createAssetTimeframeCombinations(configs: TimeframeAssetConfig[]): { asset: CryptoPair; timeframe: Timeframe }[] {
    const combinations: { asset: CryptoPair; timeframe: Timeframe }[] = [];
    
    for (const config of configs) {
      for (const timeframe of config.timeframes) {
        combinations.push({ asset: config.asset, timeframe });
      }
    }
    
    return combinations;
  }

  private getUniqueTimeframes(configs: TimeframeAssetConfig[]): Timeframe[] {
    const timeframes = new Set<Timeframe>();
    configs.forEach(config => config.timeframes.forEach(tf => timeframes.add(tf)));
    return Array.from(timeframes);
  }

  private getTimeframeMinutes(timeframe: Timeframe): number {
    const map: { [key in Timeframe]: number } = {
      '1m': 1, '3m': 3, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
    };
    return map[timeframe];
  }

  private getAssetBasePrice(asset: CryptoPair): number {
    const prices: { [key in CryptoPair]: number } = {
      'BTCUSD': 65000, 'ETHUSD': 3500, 'SOLUSD': 150
    };
    return prices[asset];
  }

  private getAssetVolatility(asset: CryptoPair): number {
    const volatilities: { [key in CryptoPair]: number } = {
      'BTCUSD': 0.02, 'ETHUSD': 0.03, 'SOLUSD': 0.05
    };
    return volatilities[asset];
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private calculateStandardDeviation(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
    
    return Math.sqrt(variance);
  }

  private determineCandleSentiment(candle: MarketDataPoint): 'BULLISH' | 'BEARISH' | 'NEUTRAL' {
    const bodySize = Math.abs(candle.close - candle.open);
    const totalRange = candle.high - candle.low;
    
    if (totalRange === 0) return 'NEUTRAL';
    
    const bodyRatio = bodySize / totalRange;
    
    if (bodyRatio > 0.6) {
      return candle.close > candle.open ? 'BULLISH' : 'BEARISH';
    }
    
    return 'NEUTRAL';
  }

  /**
   * Get supported assets
   */
  public getSupportedAssets(): CryptoPair[] {
    return [...this.supportedAssets];
  }

  /**
   * Get supported timeframes
   */
  public getSupportedTimeframes(): Timeframe[] {
    return [...this.supportedTimeframes];
  }
}

// Export factory function
export function createMultiTimeframeMultiAssetDataProvider(): MultiTimeframeMultiAssetDataProvider {
  return new MultiTimeframeMultiAssetDataProvider();
}
