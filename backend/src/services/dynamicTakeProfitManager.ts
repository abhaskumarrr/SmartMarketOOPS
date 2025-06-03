/**
 * Dynamic Take Profit Manager
 * Implements adaptive take profit system based on market conditions and momentum
 */

import { TradingSignal } from '../types/marketData';
import { logger } from '../utils/logger';

export interface MarketRegime {
  type: 'TRENDING' | 'RANGING' | 'VOLATILE';
  strength: number; // 0-100
  direction: 'UP' | 'DOWN' | 'SIDEWAYS';
  volatility: number;
  volume: number;
}

export interface TakeProfitLevel {
  percentage: number; // Percentage of position to close
  priceTarget: number; // Price target for this level
  riskRewardRatio: number; // Risk-reward ratio for this level
  trailingDistance?: number; // Trailing distance in price points
}

export interface DynamicTakeProfitConfig {
  asset: string;
  entryPrice: number;
  stopLoss: number;
  positionSize: number;
  side: 'BUY' | 'SELL';
  marketRegime: MarketRegime;
  momentum: number; // -100 to 100
  volume: number; // Volume strength multiplier
}

export class DynamicTakeProfitManager {
  private assetConfigs: Map<string, any> = new Map();

  constructor() {
    this.initializeAssetConfigs();
  }

  /**
   * Initialize asset-specific configurations based on backtest observations
   */
  private initializeAssetConfigs(): void {
    // Based on our backtest results: SOL (71% win rate), BTC (62.5%), ETH (56.5%)
    this.assetConfigs.set('BTCUSD', {
      baseVolatility: 0.025, // 2.5% average daily range
      trendingMultiplier: 1.8, // BTC trends well but moderately
      rangingMultiplier: 1.2,
      volatileMultiplier: 0.8,
      maxRiskReward: 6.0, // Conservative for BTC
      optimalWinRate: 0.625,
    });

    this.assetConfigs.set('ETHUSD', {
      baseVolatility: 0.035, // 3.5% average daily range
      trendingMultiplier: 1.6, // ETH less predictable in trends
      rangingMultiplier: 1.4,
      volatileMultiplier: 0.7,
      maxRiskReward: 5.0, // More conservative due to lower win rate
      optimalWinRate: 0.565,
    });

    this.assetConfigs.set('SOLUSD', {
      baseVolatility: 0.045, // 4.5% average daily range
      trendingMultiplier: 2.2, // SOL excellent trending performance
      rangingMultiplier: 1.8,
      volatileMultiplier: 1.0,
      maxRiskReward: 8.0, // Aggressive due to 71% win rate
      optimalWinRate: 0.71,
    });
  }

  /**
   * Generate dynamic take profit levels based on market conditions
   */
  public generateDynamicTakeProfitLevels(config: DynamicTakeProfitConfig): TakeProfitLevel[] {
    const assetConfig = this.assetConfigs.get(config.asset);
    if (!assetConfig) {
      logger.warn(`No config found for ${config.asset}, using default`);
      return this.generateDefaultTakeProfitLevels(config);
    }

    const stopLossDistance = Math.abs(config.entryPrice - config.stopLoss);
    const baseRiskReward = this.calculateBaseRiskReward(config, assetConfig);
    
    logger.info(`🎯 Generating dynamic take profit for ${config.asset}:`);
    logger.info(`   Market Regime: ${config.marketRegime.type} (${config.marketRegime.strength}%)`);
    logger.info(`   Base Risk-Reward: ${baseRiskReward.toFixed(2)}:1`);

    const levels: TakeProfitLevel[] = [];

    // Level 1: Quick profit (25% position) - Conservative
    const level1RR = Math.max(1.5, baseRiskReward * 0.4);
    const level1Target = this.calculateTargetPrice(config, stopLossDistance, level1RR);
    levels.push({
      percentage: 25,
      priceTarget: level1Target,
      riskRewardRatio: level1RR,
      trailingDistance: stopLossDistance * 0.3, // 30% of stop distance
    });

    // Level 2: Main profit (50% position) - Optimal
    const level2RR = baseRiskReward;
    const level2Target = this.calculateTargetPrice(config, stopLossDistance, level2RR);
    levels.push({
      percentage: 50,
      priceTarget: level2Target,
      riskRewardRatio: level2RR,
      trailingDistance: stopLossDistance * 0.5, // 50% of stop distance
    });

    // Level 3: Extended profit (25% position) - Aggressive
    const level3RR = Math.min(assetConfig.maxRiskReward, baseRiskReward * 1.8);
    const level3Target = this.calculateTargetPrice(config, stopLossDistance, level3RR);
    levels.push({
      percentage: 25,
      priceTarget: level3Target,
      riskRewardRatio: level3RR,
      trailingDistance: stopLossDistance * 0.8, // 80% of stop distance
    });

    // Apply momentum adjustments
    this.applyMomentumAdjustments(levels, config);

    // Apply volume adjustments
    this.applyVolumeAdjustments(levels, config);

    logger.info(`   Generated ${levels.length} take profit levels:`);
    levels.forEach((level, i) => {
      logger.info(`     Level ${i+1}: ${level.percentage}% at $${level.priceTarget.toFixed(2)} (${level.riskRewardRatio.toFixed(2)}:1)`);
    });

    return levels;
  }

  /**
   * Calculate base risk-reward ratio based on market regime and asset
   */
  private calculateBaseRiskReward(config: DynamicTakeProfitConfig, assetConfig: any): number {
    let baseRR = 3.0; // Default 3:1

    // Adjust based on market regime
    switch (config.marketRegime.type) {
      case 'TRENDING':
        baseRR = 4.0 * assetConfig.trendingMultiplier;
        break;
      case 'RANGING':
        baseRR = 2.5 * assetConfig.rangingMultiplier;
        break;
      case 'VOLATILE':
        baseRR = 3.5 * assetConfig.volatileMultiplier;
        break;
    }

    // Adjust based on regime strength
    const strengthMultiplier = 0.8 + (config.marketRegime.strength / 100) * 0.4; // 0.8 to 1.2
    baseRR *= strengthMultiplier;

    // Cap at asset maximum
    baseRR = Math.min(baseRR, assetConfig.maxRiskReward);
    
    // Ensure minimum
    baseRR = Math.max(baseRR, 1.5);

    return baseRR;
  }

  /**
   * Calculate target price based on entry, stop loss, and risk-reward ratio
   */
  private calculateTargetPrice(config: DynamicTakeProfitConfig, stopLossDistance: number, riskReward: number): number {
    const targetDistance = stopLossDistance * riskReward;
    
    if (config.side === 'BUY') {
      return config.entryPrice + targetDistance;
    } else {
      return config.entryPrice - targetDistance;
    }
  }

  /**
   * Apply momentum-based adjustments to take profit levels
   */
  private applyMomentumAdjustments(levels: TakeProfitLevel[], config: DynamicTakeProfitConfig): void {
    const momentumMultiplier = 1.0 + (config.momentum / 100) * 0.3; // ±30% adjustment
    
    levels.forEach(level => {
      if (config.side === 'BUY') {
        level.priceTarget = config.entryPrice + (level.priceTarget - config.entryPrice) * momentumMultiplier;
      } else {
        level.priceTarget = config.entryPrice - (config.entryPrice - level.priceTarget) * momentumMultiplier;
      }
      
      // Adjust trailing distance
      if (level.trailingDistance) {
        level.trailingDistance *= momentumMultiplier;
      }
    });
  }

  /**
   * Apply volume-based adjustments to take profit levels
   */
  private applyVolumeAdjustments(levels: TakeProfitLevel[], config: DynamicTakeProfitConfig): void {
    // High volume = extend targets, Low volume = tighten targets
    const volumeMultiplier = Math.max(0.7, Math.min(1.5, config.volume));
    
    // Only adjust the extended profit level (Level 3) based on volume
    if (levels.length >= 3) {
      const level3 = levels[2];
      if (config.side === 'BUY') {
        level3.priceTarget = config.entryPrice + (level3.priceTarget - config.entryPrice) * volumeMultiplier;
      } else {
        level3.priceTarget = config.entryPrice - (config.entryPrice - level3.priceTarget) * volumeMultiplier;
      }
    }
  }

  /**
   * Generate default take profit levels for unknown assets
   */
  private generateDefaultTakeProfitLevels(config: DynamicTakeProfitConfig): TakeProfitLevel[] {
    const stopLossDistance = Math.abs(config.entryPrice - config.stopLoss);
    
    return [
      {
        percentage: 25,
        priceTarget: this.calculateTargetPrice(config, stopLossDistance, 2.0),
        riskRewardRatio: 2.0,
        trailingDistance: stopLossDistance * 0.3,
      },
      {
        percentage: 50,
        priceTarget: this.calculateTargetPrice(config, stopLossDistance, 3.0),
        riskRewardRatio: 3.0,
        trailingDistance: stopLossDistance * 0.5,
      },
      {
        percentage: 25,
        priceTarget: this.calculateTargetPrice(config, stopLossDistance, 5.0),
        riskRewardRatio: 5.0,
        trailingDistance: stopLossDistance * 0.8,
      },
    ];
  }

  /**
   * Update trailing take profit levels based on current price
   */
  public updateTrailingTakeProfits(
    levels: TakeProfitLevel[],
    currentPrice: number,
    config: DynamicTakeProfitConfig
  ): TakeProfitLevel[] {
    const updatedLevels = [...levels];

    updatedLevels.forEach(level => {
      if (!level.trailingDistance) return;

      if (config.side === 'BUY') {
        // For long positions, trail up
        const potentialNewTarget = currentPrice - level.trailingDistance;
        if (potentialNewTarget > level.priceTarget) {
          level.priceTarget = potentialNewTarget;
        }
      } else {
        // For short positions, trail down
        const potentialNewTarget = currentPrice + level.trailingDistance;
        if (potentialNewTarget < level.priceTarget) {
          level.priceTarget = potentialNewTarget;
        }
      }
    });

    return updatedLevels;
  }

  /**
   * Check if any take profit levels should be executed
   */
  public checkTakeProfitExecution(
    levels: TakeProfitLevel[],
    currentPrice: number,
    config: DynamicTakeProfitConfig
  ): TakeProfitLevel[] {
    const triggeredLevels: TakeProfitLevel[] = [];

    levels.forEach(level => {
      let shouldTrigger = false;

      if (config.side === 'BUY') {
        shouldTrigger = currentPrice >= level.priceTarget;
      } else {
        shouldTrigger = currentPrice <= level.priceTarget;
      }

      if (shouldTrigger) {
        triggeredLevels.push(level);
      }
    });

    return triggeredLevels;
  }

  /**
   * Calculate breakeven stop loss after first profit target
   */
  public calculateBreakevenStop(
    config: DynamicTakeProfitConfig,
    firstProfitHit: boolean
  ): number {
    if (!firstProfitHit) {
      return config.stopLoss;
    }

    // Move stop to breakeven + small buffer (0.1% of entry price)
    const buffer = config.entryPrice * 0.001;
    
    if (config.side === 'BUY') {
      return config.entryPrice + buffer;
    } else {
      return config.entryPrice - buffer;
    }
  }

  /**
   * Get asset-specific configuration
   */
  public getAssetConfig(asset: string): any {
    return this.assetConfigs.get(asset) || {
      baseVolatility: 0.03,
      trendingMultiplier: 1.5,
      rangingMultiplier: 1.2,
      volatileMultiplier: 0.8,
      maxRiskReward: 5.0,
      optimalWinRate: 0.6,
    };
  }
}
