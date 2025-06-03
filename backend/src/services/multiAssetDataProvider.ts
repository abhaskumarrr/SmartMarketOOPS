/**
 * Multi-Asset Data Provider
 * Handles data fetching and processing for multiple cryptocurrency pairs
 */

import { MarketDataPoint, MarketDataRequest, MarketDataResponse } from '../types/marketData';
import { marketDataService } from './marketDataProvider';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { logger } from '../utils/logger';

export type CryptoPair = 'BTCUSD' | 'ETHUSD' | 'SOLUSD';

export interface AssetConfig {
  symbol: CryptoPair;
  binanceSymbol: string;
  name: string;
  category: 'large-cap' | 'mid-cap' | 'alt-coin';
  volatilityProfile: 'low' | 'medium' | 'high';
  correlationGroup: 'bitcoin' | 'ethereum' | 'layer1';
}

export interface MultiAssetData {
  timestamp: number;
  assets: {
    [key in CryptoPair]?: MarketDataPoint;
  };
  correlations: {
    btc_eth: number;
    btc_sol: number;
    eth_sol: number;
  };
  marketDominance: {
    btc: number;
    eth: number;
    sol: number;
  };
}

export interface AssetSpecificFeatures {
  symbol: CryptoPair;
  
  // Basic OHLCV
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  
  // Asset-specific volatility patterns
  volatilityRatio: number; // Compared to BTC
  volumeProfile: number; // Relative volume strength
  priceStability: number; // Price stability index
  
  // Market structure features
  supportStrength: number;
  resistanceStrength: number;
  trendConsistency: number;
  
  // Cross-asset features
  btcCorrelation: number;
  ethCorrelation: number;
  solCorrelation: number;
  
  // Asset category features
  largCapBehavior: number; // How much it behaves like large cap
  altCoinBehavior: number; // How much it behaves like alt coin
  
  // Technical indicators (asset-adjusted)
  rsi_adjusted: number; // RSI adjusted for asset volatility
  macd_strength: number; // MACD strength relative to asset
  volume_anomaly: number; // Volume anomaly detection
}

export class MultiAssetDataProvider {
  private assetConfigs: Map<CryptoPair, AssetConfig> = new Map();
  private correlationWindow = 24; // 24 hours for correlation calculation

  constructor() {
    this.initializeAssetConfigs();
  }

  /**
   * Initialize asset configurations
   */
  private initializeAssetConfigs(): void {
    const configs: AssetConfig[] = [
      {
        symbol: 'BTCUSD',
        binanceSymbol: 'BTCUSDT',
        name: 'Bitcoin',
        category: 'large-cap',
        volatilityProfile: 'medium',
        correlationGroup: 'bitcoin',
      },
      {
        symbol: 'ETHUSD',
        binanceSymbol: 'ETHUSDT',
        name: 'Ethereum',
        category: 'large-cap',
        volatilityProfile: 'medium',
        correlationGroup: 'ethereum',
      },
      {
        symbol: 'SOLUSD',
        binanceSymbol: 'SOLUSDT',
        name: 'Solana',
        category: 'alt-coin',
        volatilityProfile: 'high',
        correlationGroup: 'layer1',
      },
    ];

    configs.forEach(config => {
      this.assetConfigs.set(config.symbol, config);
    });

    logger.info('🪙 Multi-asset configurations initialized', {
      assets: configs.map(c => `${c.symbol} (${c.category}, ${c.volatilityProfile})`),
    });
  }

  /**
   * Fetch historical data for all supported assets
   */
  public async fetchMultiAssetData(
    timeframe: string,
    startDate: Date,
    endDate: Date,
    assets: CryptoPair[] = ['BTCUSD', 'ETHUSD', 'SOLUSD']
  ): Promise<{ [key in CryptoPair]?: MarketDataPoint[] }> {
    
    logger.info('📊 Fetching multi-asset data...', {
      assets,
      timeframe,
      period: `${startDate.toISOString().split('T')[0]} to ${endDate.toISOString().split('T')[0]}`,
    });

    const assetData: { [key in CryptoPair]?: MarketDataPoint[] } = {};
    const fetchPromises: Promise<void>[] = [];

    // Fetch data for each asset in parallel
    assets.forEach(asset => {
      const promise = this.fetchAssetData(asset, timeframe, startDate, endDate)
        .then(data => {
          assetData[asset] = data;
          logger.info(`✅ Fetched ${data.length} candles for ${asset}`);
        })
        .catch(error => {
          logger.error(`❌ Failed to fetch data for ${asset}:`, error);
          assetData[asset] = [];
        });
      
      fetchPromises.push(promise);
    });

    await Promise.all(fetchPromises);

    const totalCandles = Object.values(assetData).reduce((sum, data) => sum + (data?.length || 0), 0);
    logger.info('✅ Multi-asset data fetching completed', {
      totalCandles,
      assetsLoaded: Object.keys(assetData).length,
    });

    return assetData;
  }

  /**
   * Fetch data for a specific asset
   */
  private async fetchAssetData(
    asset: CryptoPair,
    timeframe: string,
    startDate: Date,
    endDate: Date
  ): Promise<MarketDataPoint[]> {
    
    try {
      const response = await marketDataService.fetchHistoricalData({
        symbol: asset,
        timeframe,
        startDate,
        endDate,
        exchange: 'binance',
      }, 'binance');

      return response.data;
    } catch (error) {
      logger.warn(`⚠️ Failed to fetch real data for ${asset}, using mock data`);
      
      // Fallback to enhanced mock data
      const response = await marketDataService.fetchHistoricalData({
        symbol: asset,
        timeframe,
        startDate,
        endDate,
        exchange: 'enhanced-mock',
      }, 'enhanced-mock');

      return response.data;
    }
  }

  /**
   * Generate asset-specific features for model training
   */
  public generateAssetSpecificFeatures(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    targetAsset: CryptoPair,
    index: number
  ): AssetSpecificFeatures | null {
    
    const data = assetData[targetAsset];
    if (!data || index >= data.length) {
      return null;
    }

    const currentCandle = data[index];
    const config = this.assetConfigs.get(targetAsset);
    
    if (!config) {
      return null;
    }

    // Calculate asset-specific volatility
    const volatilityRatio = this.calculateVolatilityRatio(assetData, targetAsset, index);
    
    // Calculate volume profile
    const volumeProfile = this.calculateVolumeProfile(data, index);
    
    // Calculate price stability
    const priceStability = this.calculatePriceStability(data, index);
    
    // Calculate market structure
    const { supportStrength, resistanceStrength } = this.calculateMarketStructure(data, index);
    const trendConsistency = this.calculateTrendConsistency(data, index);
    
    // Calculate cross-asset correlations
    const correlations = this.calculateCrossAssetCorrelations(assetData, targetAsset, index);
    
    // Calculate asset category behaviors
    const { largCapBehavior, altCoinBehavior } = this.calculateCategoryBehaviors(config, data, index);
    
    // Calculate adjusted technical indicators
    const rsiAdjusted = this.calculateAdjustedRSI(data, index, config.volatilityProfile);
    const macdStrength = this.calculateMACDStrength(data, index, config.volatilityProfile);
    const volumeAnomaly = this.calculateVolumeAnomaly(data, index);

    return {
      symbol: targetAsset,
      
      // Basic OHLCV
      open: currentCandle.open,
      high: currentCandle.high,
      low: currentCandle.low,
      close: currentCandle.close,
      volume: currentCandle.volume,
      
      // Asset-specific features
      volatilityRatio,
      volumeProfile,
      priceStability,
      
      // Market structure
      supportStrength,
      resistanceStrength,
      trendConsistency,
      
      // Cross-asset correlations
      btcCorrelation: correlations.btc,
      ethCorrelation: correlations.eth,
      solCorrelation: correlations.sol,
      
      // Category behaviors
      largCapBehavior,
      altCoinBehavior,
      
      // Adjusted indicators
      rsi_adjusted: rsiAdjusted,
      macd_strength: macdStrength,
      volume_anomaly: volumeAnomaly,
    };
  }

  /**
   * Calculate volatility ratio compared to BTC
   */
  private calculateVolatilityRatio(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    targetAsset: CryptoPair,
    index: number
  ): number {
    
    if (targetAsset === 'BTCUSD') return 1.0;
    
    const targetData = assetData[targetAsset];
    const btcData = assetData['BTCUSD'];
    
    if (!targetData || !btcData || index < 10) return 1.0;
    
    const targetVolatility = this.calculateRecentVolatility(targetData, index, 10);
    const btcVolatility = this.calculateRecentVolatility(btcData, index, 10);
    
    return btcVolatility > 0 ? targetVolatility / btcVolatility : 1.0;
  }

  /**
   * Calculate recent volatility for an asset
   */
  private calculateRecentVolatility(data: MarketDataPoint[], index: number, period: number): number {
    if (index < period) return 0;
    
    const recentPrices = data.slice(index - period + 1, index + 1).map(d => d.close);
    const returns = recentPrices.slice(1).map((price, i) => Math.log(price / recentPrices[i]));
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate volume profile strength
   */
  private calculateVolumeProfile(data: MarketDataPoint[], index: number): number {
    if (index < 20) return 1.0;
    
    const currentVolume = data[index].volume;
    const recentVolumes = data.slice(index - 19, index).map(d => d.volume);
    const avgVolume = recentVolumes.reduce((sum, vol) => sum + vol, 0) / recentVolumes.length;
    
    return avgVolume > 0 ? currentVolume / avgVolume : 1.0;
  }

  /**
   * Calculate price stability index
   */
  private calculatePriceStability(data: MarketDataPoint[], index: number): number {
    if (index < 10) return 0.5;
    
    const recentPrices = data.slice(index - 9, index + 1).map(d => d.close);
    const priceChanges = recentPrices.slice(1).map((price, i) => Math.abs(price - recentPrices[i]) / recentPrices[i]);
    const avgChange = priceChanges.reduce((sum, change) => sum + change, 0) / priceChanges.length;
    
    // Invert so higher values mean more stability
    return Math.max(0, 1 - avgChange * 10);
  }

  /**
   * Calculate market structure strength
   */
  private calculateMarketStructure(data: MarketDataPoint[], index: number): { supportStrength: number; resistanceStrength: number } {
    if (index < 50) return { supportStrength: 0.5, resistanceStrength: 0.5 };
    
    const recentData = data.slice(index - 49, index + 1);
    const currentPrice = data[index].close;
    
    // Find support and resistance levels
    const lows = recentData.map(d => d.low);
    const highs = recentData.map(d => d.high);
    
    const supportLevel = Math.min(...lows);
    const resistanceLevel = Math.max(...highs);
    
    // Calculate strength based on how many times price tested these levels
    const supportTests = recentData.filter(d => Math.abs(d.low - supportLevel) / supportLevel < 0.01).length;
    const resistanceTests = recentData.filter(d => Math.abs(d.high - resistanceLevel) / resistanceLevel < 0.01).length;
    
    const supportStrength = Math.min(1, supportTests / 5); // Normalize to 0-1
    const resistanceStrength = Math.min(1, resistanceTests / 5);
    
    return { supportStrength, resistanceStrength };
  }

  /**
   * Calculate trend consistency
   */
  private calculateTrendConsistency(data: MarketDataPoint[], index: number): number {
    if (index < 20) return 0.5;
    
    const recentPrices = data.slice(index - 19, index + 1).map(d => d.close);
    let consistentMoves = 0;
    let totalMoves = 0;
    
    for (let i = 1; i < recentPrices.length; i++) {
      const currentMove = recentPrices[i] > recentPrices[i - 1] ? 1 : -1;
      const overallTrend = recentPrices[recentPrices.length - 1] > recentPrices[0] ? 1 : -1;
      
      if (currentMove === overallTrend) {
        consistentMoves++;
      }
      totalMoves++;
    }
    
    return totalMoves > 0 ? consistentMoves / totalMoves : 0.5;
  }

  /**
   * Calculate cross-asset correlations
   */
  private calculateCrossAssetCorrelations(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    targetAsset: CryptoPair,
    index: number
  ): { btc: number; eth: number; sol: number } {
    
    const defaultCorrelations = { btc: 0, eth: 0, sol: 0 };
    
    if (index < this.correlationWindow) {
      return defaultCorrelations;
    }

    const targetData = assetData[targetAsset];
    if (!targetData) return defaultCorrelations;

    const correlations = { btc: 0, eth: 0, sol: 0 };
    
    // Calculate correlation with each asset
    (['BTCUSD', 'ETHUSD', 'SOLUSD'] as CryptoPair[]).forEach(asset => {
      if (asset === targetAsset) {
        correlations[asset.substring(0, 3).toLowerCase() as keyof typeof correlations] = 1.0;
        return;
      }
      
      const otherData = assetData[asset];
      if (otherData && otherData.length > index) {
        const correlation = this.calculatePearsonCorrelation(targetData, otherData, index, this.correlationWindow);
        correlations[asset.substring(0, 3).toLowerCase() as keyof typeof correlations] = correlation;
      }
    });

    return correlations;
  }

  /**
   * Calculate Pearson correlation between two assets
   */
  private calculatePearsonCorrelation(
    data1: MarketDataPoint[],
    data2: MarketDataPoint[],
    index: number,
    window: number
  ): number {
    
    if (index < window || data1.length <= index || data2.length <= index) {
      return 0;
    }

    const returns1 = this.calculateReturns(data1, index - window + 1, index + 1);
    const returns2 = this.calculateReturns(data2, index - window + 1, index + 1);
    
    if (returns1.length !== returns2.length || returns1.length === 0) {
      return 0;
    }

    const mean1 = returns1.reduce((sum, ret) => sum + ret, 0) / returns1.length;
    const mean2 = returns2.reduce((sum, ret) => sum + ret, 0) / returns2.length;
    
    let numerator = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;
    
    for (let i = 0; i < returns1.length; i++) {
      const diff1 = returns1[i] - mean1;
      const diff2 = returns2[i] - mean2;
      
      numerator += diff1 * diff2;
      sumSq1 += diff1 * diff1;
      sumSq2 += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(sumSq1 * sumSq2);
    return denominator > 0 ? numerator / denominator : 0;
  }

  /**
   * Calculate returns for a data slice
   */
  private calculateReturns(data: MarketDataPoint[], startIndex: number, endIndex: number): number[] {
    const slice = data.slice(startIndex, endIndex);
    const returns: number[] = [];
    
    for (let i = 1; i < slice.length; i++) {
      const ret = Math.log(slice[i].close / slice[i - 1].close);
      returns.push(ret);
    }
    
    return returns;
  }

  /**
   * Calculate asset category behaviors
   */
  private calculateCategoryBehaviors(
    config: AssetConfig,
    data: MarketDataPoint[],
    index: number
  ): { largCapBehavior: number; altCoinBehavior: number } {
    
    // Base behavior on asset category
    let largCapBehavior = 0.5;
    let altCoinBehavior = 0.5;
    
    if (config.category === 'large-cap') {
      largCapBehavior = 0.8;
      altCoinBehavior = 0.2;
    } else if (config.category === 'alt-coin') {
      largCapBehavior = 0.2;
      altCoinBehavior = 0.8;
    }
    
    // Adjust based on recent volatility
    if (index >= 10) {
      const recentVolatility = this.calculateRecentVolatility(data, index, 10);
      
      // High volatility increases alt-coin behavior
      if (recentVolatility > 0.05) {
        altCoinBehavior = Math.min(1, altCoinBehavior + 0.2);
        largCapBehavior = Math.max(0, largCapBehavior - 0.2);
      }
    }
    
    return { largCapBehavior, altCoinBehavior };
  }

  /**
   * Calculate adjusted RSI for asset volatility
   */
  private calculateAdjustedRSI(data: MarketDataPoint[], index: number, volatilityProfile: string): number {
    if (index < 14) return 50;
    
    const closes = data.slice(Math.max(0, index - 13), index + 1).map(d => d.close);
    const rsi = technicalAnalysis.calculateRSI(closes, 14);
    const baseRSI = rsi[rsi.length - 1] || 50;
    
    // Adjust RSI based on volatility profile
    let adjustment = 0;
    if (volatilityProfile === 'high') {
      // For high volatility assets, moderate extreme RSI values
      if (baseRSI > 70) adjustment = (baseRSI - 70) * -0.3;
      else if (baseRSI < 30) adjustment = (30 - baseRSI) * 0.3;
    }
    
    return Math.max(0, Math.min(100, baseRSI + adjustment));
  }

  /**
   * Calculate MACD strength relative to asset
   */
  private calculateMACDStrength(data: MarketDataPoint[], index: number, volatilityProfile: string): number {
    if (index < 26) return 0;
    
    const closes = data.slice(Math.max(0, index - 25), index + 1).map(d => d.close);
    const macd = technicalAnalysis.calculateMACD(closes, 12, 26, 9);
    const macdValue = macd.macd[macd.macd.length - 1] || 0;
    
    // Normalize MACD based on recent price levels
    const recentPrice = closes[closes.length - 1];
    const normalizedMACD = macdValue / recentPrice;
    
    // Adjust for volatility profile
    const volatilityMultiplier = volatilityProfile === 'high' ? 0.5 : volatilityProfile === 'low' ? 1.5 : 1.0;
    
    return normalizedMACD * volatilityMultiplier;
  }

  /**
   * Calculate volume anomaly detection
   */
  private calculateVolumeAnomaly(data: MarketDataPoint[], index: number): number {
    if (index < 20) return 0;
    
    const currentVolume = data[index].volume;
    const recentVolumes = data.slice(index - 19, index).map(d => d.volume);
    
    const mean = recentVolumes.reduce((sum, vol) => sum + vol, 0) / recentVolumes.length;
    const variance = recentVolumes.reduce((sum, vol) => sum + Math.pow(vol - mean, 2), 0) / recentVolumes.length;
    const stdDev = Math.sqrt(variance);
    
    if (stdDev === 0) return 0;
    
    // Z-score for volume anomaly
    const zScore = (currentVolume - mean) / stdDev;
    
    // Return normalized anomaly score (0 = normal, 1 = high anomaly)
    return Math.min(1, Math.abs(zScore) / 3);
  }

  /**
   * Get asset configuration
   */
  public getAssetConfig(asset: CryptoPair): AssetConfig | undefined {
    return this.assetConfigs.get(asset);
  }

  /**
   * Get all supported assets
   */
  public getSupportedAssets(): CryptoPair[] {
    return Array.from(this.assetConfigs.keys());
  }
}

// Export factory function
export function createMultiAssetDataProvider(): MultiAssetDataProvider {
  return new MultiAssetDataProvider();
}
