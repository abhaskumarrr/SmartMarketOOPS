/**
 * Multi-Asset Model Training Data Processor
 * Processes training data for multiple cryptocurrency pairs with cross-asset features
 */

import { TrainingFeatures, TrainingDataset } from './modelTrainingDataProcessor';
import { createMultiAssetDataProvider, CryptoPair, AssetSpecificFeatures } from './multiAssetDataProvider';
import { MarketDataPoint } from '../types/marketData';
import { logger } from '../utils/logger';

export interface MultiAssetTrainingFeatures extends TrainingFeatures {
  // Asset identification
  asset_type: number; // 0=BTC, 1=ETH, 2=SOL
  
  // Cross-asset features
  btc_price_normalized: number;
  eth_price_normalized: number;
  sol_price_normalized: number;
  
  // Cross-asset correlations
  btc_correlation: number;
  eth_correlation: number;
  sol_correlation: number;
  
  // Market dominance features
  btc_dominance: number;
  eth_dominance: number;
  sol_dominance: number;
  
  // Asset-specific volatility
  volatility_ratio: number;
  volume_profile: number;
  price_stability: number;
  
  // Market structure (asset-specific)
  support_strength: number;
  resistance_strength: number;
  trend_consistency: number;
  
  // Category behaviors
  large_cap_behavior: number;
  alt_coin_behavior: number;
  
  // Adjusted technical indicators
  rsi_adjusted: number;
  macd_strength: number;
  volume_anomaly: number;
  
  // Cross-asset momentum
  cross_asset_momentum: number;
  relative_strength: number;
  
  // Multi-asset targets
  btc_future_return_1h: number;
  eth_future_return_1h: number;
  sol_future_return_1h: number;
  
  portfolio_future_return_1h: number;
  best_asset_1h: number; // 0=BTC, 1=ETH, 2=SOL
}

export interface MultiAssetTrainingDataset {
  features: MultiAssetTrainingFeatures[];
  assetBreakdown: {
    btc: number;
    eth: number;
    sol: number;
  };
  correlationStats: {
    btc_eth: number;
    btc_sol: number;
    eth_sol: number;
  };
  metadata: {
    symbols: CryptoPair[];
    startDate: Date;
    endDate: Date;
    totalSamples: number;
    featureCount: number;
    trainSplit: number;
    validationSplit: number;
    testSplit: number;
  };
}

export class MultiAssetModelTrainer {
  private dataProvider = createMultiAssetDataProvider();

  /**
   * Process multi-asset training data
   */
  public async processMultiAssetTrainingData(
    timeframe: string,
    startDate: Date,
    endDate: Date,
    assets: CryptoPair[] = ['BTCUSD', 'ETHUSD', 'SOLUSD'],
    trainSplit: number = 0.7,
    validationSplit: number = 0.15,
    testSplit: number = 0.15
  ): Promise<MultiAssetTrainingDataset> {
    
    logger.info('🪙 Processing multi-asset training data...', {
      assets,
      timeframe,
      period: `${startDate.toISOString().split('T')[0]} to ${endDate.toISOString().split('T')[0]}`,
    });

    // Fetch data for all assets
    const assetData = await this.dataProvider.fetchMultiAssetData(timeframe, startDate, endDate, assets);
    
    // Validate data availability
    this.validateAssetData(assetData, assets);
    
    // Generate unified training features
    const features = this.generateUnifiedFeatures(assetData, assets);
    
    // Calculate cross-asset statistics
    const correlationStats = this.calculateCorrelationStats(features);
    const assetBreakdown = this.calculateAssetBreakdown(features);
    
    // Clean and validate features
    const cleanFeatures = this.cleanMultiAssetData(features);
    
    logger.info('✅ Multi-asset training data processed', {
      totalSamples: cleanFeatures.length,
      assetBreakdown,
      correlationStats,
      featureCount: Object.keys(cleanFeatures[0] || {}).length,
    });

    return {
      features: cleanFeatures,
      assetBreakdown,
      correlationStats,
      metadata: {
        symbols: assets,
        startDate,
        endDate,
        totalSamples: cleanFeatures.length,
        featureCount: Object.keys(cleanFeatures[0] || {}).length,
        trainSplit,
        validationSplit,
        testSplit,
      },
    };
  }

  /**
   * Validate asset data availability
   */
  private validateAssetData(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    assets: CryptoPair[]
  ): void {
    
    assets.forEach(asset => {
      const data = assetData[asset];
      if (!data || data.length === 0) {
        throw new Error(`No data available for ${asset}`);
      }
      
      logger.info(`📊 ${asset}: ${data.length} candles loaded`);
    });

    // Check data alignment
    const dataLengths = assets.map(asset => assetData[asset]?.length || 0);
    const minLength = Math.min(...dataLengths);
    const maxLength = Math.max(...dataLengths);
    
    if (maxLength - minLength > dataLengths[0] * 0.1) {
      logger.warn('⚠️ Significant data length differences detected between assets', {
        lengths: assets.map((asset, i) => ({ asset, length: dataLengths[i] })),
      });
    }
  }

  /**
   * Generate unified features combining all assets
   */
  private generateUnifiedFeatures(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    assets: CryptoPair[]
  ): Partial<MultiAssetTrainingFeatures>[] {
    
    const allFeatures: Partial<MultiAssetTrainingFeatures>[] = [];
    
    // Find the minimum data length to ensure alignment
    const minLength = Math.min(...assets.map(asset => assetData[asset]?.length || 0));
    
    // Process each time point
    for (let i = 50; i < minLength - 24; i++) { // Leave buffer for indicators and future returns
      
      // Generate features for each asset at this time point
      assets.forEach(asset => {
        const assetFeatures = this.generateAssetFeatures(assetData, asset, i, assets);
        if (assetFeatures) {
          allFeatures.push(assetFeatures);
        }
      });
    }
    
    return allFeatures;
  }

  /**
   * Generate features for a specific asset at a specific time point
   */
  private generateAssetFeatures(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    targetAsset: CryptoPair,
    index: number,
    allAssets: CryptoPair[]
  ): Partial<MultiAssetTrainingFeatures> | null {
    
    const data = assetData[targetAsset];
    if (!data || index >= data.length) {
      return null;
    }

    const currentCandle = data[index];
    
    // Get asset-specific features
    const assetSpecific = this.dataProvider.generateAssetSpecificFeatures(assetData, targetAsset, index);
    if (!assetSpecific) {
      return null;
    }

    // Calculate cross-asset features
    const crossAssetFeatures = this.calculateCrossAssetFeatures(assetData, targetAsset, index, allAssets);
    
    // Calculate future returns for all assets
    const futureReturns = this.calculateMultiAssetFutureReturns(assetData, index, allAssets);
    
    // Determine best performing asset in next hour
    const bestAsset = this.determineBestAsset(futureReturns);
    
    // Calculate portfolio return (equal weight)
    const portfolioReturn = (futureReturns.btc + futureReturns.eth + futureReturns.sol) / 3;

    return {
      // Asset identification
      asset_type: this.getAssetTypeNumber(targetAsset),
      
      // Basic OHLCV from asset-specific features
      open: assetSpecific.open,
      high: assetSpecific.high,
      low: assetSpecific.low,
      close: assetSpecific.close,
      volume: assetSpecific.volume,
      
      // Technical indicators (simplified from base features)
      rsi_14: assetSpecific.rsi_adjusted,
      ema_12: currentCandle.close * 0.99, // Simplified
      ema_26: currentCandle.close * 1.01, // Simplified
      macd: assetSpecific.macd_strength,
      volume_sma_20: assetSpecific.volume * 0.8, // Simplified
      
      // Cross-asset normalized prices
      btc_price_normalized: crossAssetFeatures.btc_price_normalized,
      eth_price_normalized: crossAssetFeatures.eth_price_normalized,
      sol_price_normalized: crossAssetFeatures.sol_price_normalized,
      
      // Cross-asset correlations
      btc_correlation: assetSpecific.btcCorrelation,
      eth_correlation: assetSpecific.ethCorrelation,
      sol_correlation: assetSpecific.solCorrelation,
      
      // Market dominance (simplified calculation)
      btc_dominance: crossAssetFeatures.btc_dominance,
      eth_dominance: crossAssetFeatures.eth_dominance,
      sol_dominance: crossAssetFeatures.sol_dominance,
      
      // Asset-specific features
      volatility_ratio: assetSpecific.volatilityRatio,
      volume_profile: assetSpecific.volumeProfile,
      price_stability: assetSpecific.priceStability,
      
      // Market structure
      support_strength: assetSpecific.supportStrength,
      resistance_strength: assetSpecific.resistanceStrength,
      trend_consistency: assetSpecific.trendConsistency,
      
      // Category behaviors
      large_cap_behavior: assetSpecific.largCapBehavior,
      alt_coin_behavior: assetSpecific.altCoinBehavior,
      
      // Adjusted indicators
      rsi_adjusted: assetSpecific.rsi_adjusted,
      macd_strength: assetSpecific.macd_strength,
      volume_anomaly: assetSpecific.volume_anomaly,
      
      // Cross-asset momentum
      cross_asset_momentum: crossAssetFeatures.cross_asset_momentum,
      relative_strength: crossAssetFeatures.relative_strength,
      
      // Time features
      hour_of_day: new Date(currentCandle.timestamp).getUTCHours() / 23,
      day_of_week: new Date(currentCandle.timestamp).getUTCDay() / 6,
      
      // Future returns (targets)
      future_return_1h: futureReturns[targetAsset.substring(0, 3).toLowerCase() as keyof typeof futureReturns] || 0,
      btc_future_return_1h: futureReturns.btc,
      eth_future_return_1h: futureReturns.eth,
      sol_future_return_1h: futureReturns.sol,
      portfolio_future_return_1h: portfolioReturn,
      best_asset_1h: bestAsset,
      
      // Classification signals
      signal_1h: this.returnToSignal(futureReturns[targetAsset.substring(0, 3).toLowerCase() as keyof typeof futureReturns] || 0),
    };
  }

  /**
   * Calculate cross-asset features
   */
  private calculateCrossAssetFeatures(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    targetAsset: CryptoPair,
    index: number,
    allAssets: CryptoPair[]
  ): any {
    
    const prices: { [key: string]: number } = {};
    let totalMarketCap = 0;
    
    // Get current prices and calculate total market cap (simplified)
    allAssets.forEach(asset => {
      const data = assetData[asset];
      if (data && index < data.length) {
        const price = data[index].close;
        const volume = data[index].volume;
        prices[asset] = price;
        totalMarketCap += price * volume; // Simplified market cap
      }
    });
    
    // Normalize prices (0-1 scale based on recent range)
    const normalizedPrices = this.normalizePrices(assetData, index, allAssets);
    
    // Calculate market dominance
    const btcMarketCap = (prices['BTCUSD'] || 0) * (assetData['BTCUSD']?.[index]?.volume || 0);
    const ethMarketCap = (prices['ETHUSD'] || 0) * (assetData['ETHUSD']?.[index]?.volume || 0);
    const solMarketCap = (prices['SOLUSD'] || 0) * (assetData['SOLUSD']?.[index]?.volume || 0);
    
    const btcDominance = totalMarketCap > 0 ? btcMarketCap / totalMarketCap : 0.33;
    const ethDominance = totalMarketCap > 0 ? ethMarketCap / totalMarketCap : 0.33;
    const solDominance = totalMarketCap > 0 ? solMarketCap / totalMarketCap : 0.33;
    
    // Calculate cross-asset momentum
    const crossAssetMomentum = this.calculateCrossAssetMomentum(assetData, index, allAssets);
    
    // Calculate relative strength
    const relativeStrength = this.calculateRelativeStrength(assetData, targetAsset, index, allAssets);

    return {
      btc_price_normalized: normalizedPrices.btc,
      eth_price_normalized: normalizedPrices.eth,
      sol_price_normalized: normalizedPrices.sol,
      btc_dominance: btcDominance,
      eth_dominance: ethDominance,
      sol_dominance: solDominance,
      cross_asset_momentum: crossAssetMomentum,
      relative_strength: relativeStrength,
    };
  }

  /**
   * Normalize prices based on recent range
   */
  private normalizePrices(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    index: number,
    allAssets: CryptoPair[]
  ): { btc: number; eth: number; sol: number } {
    
    const normalized = { btc: 0.5, eth: 0.5, sol: 0.5 };
    const lookback = 100; // 100 periods for normalization
    
    allAssets.forEach(asset => {
      const data = assetData[asset];
      if (data && index >= lookback) {
        const recentData = data.slice(index - lookback, index + 1);
        const prices = recentData.map(d => d.close);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const currentPrice = data[index].close;
        
        const normalizedValue = maxPrice > minPrice ? (currentPrice - minPrice) / (maxPrice - minPrice) : 0.5;
        
        const assetKey = asset.substring(0, 3).toLowerCase() as keyof typeof normalized;
        normalized[assetKey] = normalizedValue;
      }
    });
    
    return normalized;
  }

  /**
   * Calculate cross-asset momentum
   */
  private calculateCrossAssetMomentum(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    index: number,
    allAssets: CryptoPair[]
  ): number {
    
    if (index < 10) return 0;
    
    let totalMomentum = 0;
    let assetCount = 0;
    
    allAssets.forEach(asset => {
      const data = assetData[asset];
      if (data && index < data.length) {
        const currentPrice = data[index].close;
        const pastPrice = data[index - 10].close;
        const momentum = (currentPrice - pastPrice) / pastPrice;
        totalMomentum += momentum;
        assetCount++;
      }
    });
    
    return assetCount > 0 ? totalMomentum / assetCount : 0;
  }

  /**
   * Calculate relative strength of target asset
   */
  private calculateRelativeStrength(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    targetAsset: CryptoPair,
    index: number,
    allAssets: CryptoPair[]
  ): number {
    
    if (index < 20) return 0.5;
    
    const targetData = assetData[targetAsset];
    if (!targetData) return 0.5;
    
    const targetReturn = (targetData[index].close - targetData[index - 20].close) / targetData[index - 20].close;
    
    let avgReturn = 0;
    let assetCount = 0;
    
    allAssets.forEach(asset => {
      if (asset !== targetAsset) {
        const data = assetData[asset];
        if (data && index < data.length) {
          const assetReturn = (data[index].close - data[index - 20].close) / data[index - 20].close;
          avgReturn += assetReturn;
          assetCount++;
        }
      }
    });
    
    if (assetCount === 0) return 0.5;
    
    avgReturn /= assetCount;
    
    // Return relative strength (0.5 = average, >0.5 = outperforming, <0.5 = underperforming)
    return 0.5 + (targetReturn - avgReturn);
  }

  /**
   * Calculate future returns for all assets
   */
  private calculateMultiAssetFutureReturns(
    assetData: { [key in CryptoPair]?: MarketDataPoint[] },
    index: number,
    allAssets: CryptoPair[]
  ): { btc: number; eth: number; sol: number } {
    
    const returns = { btc: 0, eth: 0, sol: 0 };
    
    allAssets.forEach(asset => {
      const data = assetData[asset];
      if (data && index + 1 < data.length) {
        const currentPrice = data[index].close;
        const futurePrice = data[index + 1].close;
        const futureReturn = (futurePrice - currentPrice) / currentPrice;
        
        const assetKey = asset.substring(0, 3).toLowerCase() as keyof typeof returns;
        returns[assetKey] = futureReturn;
      }
    });
    
    return returns;
  }

  /**
   * Determine best performing asset
   */
  private determineBestAsset(returns: { btc: number; eth: number; sol: number }): number {
    const assets = [
      { name: 'btc', return: returns.btc, index: 0 },
      { name: 'eth', return: returns.eth, index: 1 },
      { name: 'sol', return: returns.sol, index: 2 },
    ];
    
    const bestAsset = assets.reduce((best, current) => 
      current.return > best.return ? current : best
    );
    
    return bestAsset.index;
  }

  /**
   * Get asset type number
   */
  private getAssetTypeNumber(asset: CryptoPair): number {
    switch (asset) {
      case 'BTCUSD': return 0;
      case 'ETHUSD': return 1;
      case 'SOLUSD': return 2;
      default: return 0;
    }
  }

  /**
   * Convert return to signal
   */
  private returnToSignal(returnValue: number): number {
    if (returnValue > 0.005) return 1; // Buy signal for >0.5% return
    if (returnValue < -0.005) return -1; // Sell signal for <-0.5% return
    return 0; // Hold signal
  }

  /**
   * Calculate correlation statistics
   */
  private calculateCorrelationStats(features: Partial<MultiAssetTrainingFeatures>[]): any {
    if (features.length === 0) {
      return { btc_eth: 0, btc_sol: 0, eth_sol: 0 };
    }

    const correlations = features.map(f => ({
      btc_eth: f.btc_correlation || 0,
      btc_sol: f.btc_correlation || 0,
      eth_sol: f.eth_correlation || 0,
    }));

    return {
      btc_eth: correlations.reduce((sum, c) => sum + c.btc_eth, 0) / correlations.length,
      btc_sol: correlations.reduce((sum, c) => sum + c.btc_sol, 0) / correlations.length,
      eth_sol: correlations.reduce((sum, c) => sum + c.eth_sol, 0) / correlations.length,
    };
  }

  /**
   * Calculate asset breakdown
   */
  private calculateAssetBreakdown(features: Partial<MultiAssetTrainingFeatures>[]): any {
    const breakdown = { btc: 0, eth: 0, sol: 0 };
    
    features.forEach(f => {
      if (f.asset_type === 0) breakdown.btc++;
      else if (f.asset_type === 1) breakdown.eth++;
      else if (f.asset_type === 2) breakdown.sol++;
    });
    
    return breakdown;
  }

  /**
   * Clean multi-asset data
   */
  private cleanMultiAssetData(features: Partial<MultiAssetTrainingFeatures>[]): MultiAssetTrainingFeatures[] {
    return features.filter((feature, index) => {
      // Remove samples with NaN or undefined values
      const values = Object.values(feature);
      const hasInvalidValues = values.some(value => 
        value === null || value === undefined || isNaN(value as number)
      );
      
      return !hasInvalidValues;
    }) as MultiAssetTrainingFeatures[];
  }

  /**
   * Split multi-asset dataset
   */
  public splitMultiAssetDataset(dataset: MultiAssetTrainingDataset): {
    train: MultiAssetTrainingFeatures[];
    validation: MultiAssetTrainingFeatures[];
    test: MultiAssetTrainingFeatures[];
  } {
    const { features, metadata } = dataset;
    const totalSamples = features.length;
    
    const trainSize = Math.floor(totalSamples * metadata.trainSplit);
    const validationSize = Math.floor(totalSamples * metadata.validationSplit);
    
    const train = features.slice(0, trainSize);
    const validation = features.slice(trainSize, trainSize + validationSize);
    const test = features.slice(trainSize + validationSize);
    
    logger.info('📊 Multi-asset dataset split completed', {
      total: totalSamples,
      train: train.length,
      validation: validation.length,
      test: test.length,
      assetBreakdown: metadata.symbols,
    });

    return { train, validation, test };
  }
}

// Export factory function
export function createMultiAssetModelTrainer(): MultiAssetModelTrainer {
  return new MultiAssetModelTrainer();
}
