/**
 * Dynamic Take Profit Manager
 * Implements adaptive take profit system based on market conditions and momentum
 */
export interface MarketRegime {
    type: 'TRENDING' | 'RANGING' | 'VOLATILE';
    strength: number;
    direction: 'UP' | 'DOWN' | 'SIDEWAYS';
    volatility: number;
    volume: number;
}
export interface TakeProfitLevel {
    percentage: number;
    priceTarget: number;
    riskRewardRatio: number;
    trailingDistance?: number;
}
export interface DynamicTakeProfitConfig {
    asset: string;
    entryPrice: number;
    stopLoss: number;
    positionSize: number;
    side: 'BUY' | 'SELL';
    marketRegime: MarketRegime;
    momentum: number;
    volume: number;
}
export declare class DynamicTakeProfitManager {
    private assetConfigs;
    constructor();
    /**
     * Initialize asset-specific configurations based on backtest observations
     */
    private initializeAssetConfigs;
    /**
     * Generate dynamic take profit levels based on market conditions
     */
    generateDynamicTakeProfitLevels(config: DynamicTakeProfitConfig): TakeProfitLevel[];
    /**
     * Calculate base risk-reward ratio based on market regime and asset
     */
    private calculateBaseRiskReward;
    /**
     * Calculate target price based on entry, stop loss, and risk-reward ratio
     */
    private calculateTargetPrice;
    /**
     * Apply momentum-based adjustments to take profit levels
     */
    private applyMomentumAdjustments;
    /**
     * Apply volume-based adjustments to take profit levels
     */
    private applyVolumeAdjustments;
    /**
     * Generate default take profit levels for unknown assets
     */
    private generateDefaultTakeProfitLevels;
    /**
     * Update trailing take profit levels based on current price
     */
    updateTrailingTakeProfits(levels: TakeProfitLevel[], currentPrice: number, config: DynamicTakeProfitConfig): TakeProfitLevel[];
    /**
     * Check if any take profit levels should be executed
     */
    checkTakeProfitExecution(levels: TakeProfitLevel[], currentPrice: number, config: DynamicTakeProfitConfig): TakeProfitLevel[];
    /**
     * Calculate breakeven stop loss after first profit target
     */
    calculateBreakevenStop(config: DynamicTakeProfitConfig, firstProfitHit: boolean): number;
    /**
     * Get asset-specific configuration
     */
    getAssetConfig(asset: string): any;
}
