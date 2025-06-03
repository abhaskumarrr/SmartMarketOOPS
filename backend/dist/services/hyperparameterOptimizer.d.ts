/**
 * Hyperparameter Optimization Engine
 * Systematically tests different parameter combinations to maximize trading performance
 */
export interface OptimizationRanges {
    minConfidence: number[];
    modelConsensus: number[];
    decisionCooldown: number[];
    riskPerTrade: number[];
    stopLossPercent: number[];
    takeProfitMultiplier: number[];
    positionSizeMultiplier: number[];
    trendThreshold: number[];
    volatilityThreshold: number[];
}
export interface ParameterConfig {
    id: string;
    minConfidence: number;
    modelConsensus: number;
    decisionCooldown: number;
    riskPerTrade: number;
    stopLossPercent: number;
    takeProfitMultiplier: number;
    positionSizeMultiplier: number;
    trendThreshold: number;
    volatilityThreshold: number;
}
export interface OptimizationResult {
    config: ParameterConfig;
    performance: {
        totalReturnPercent: number;
        sharpeRatio: number;
        maxDrawdownPercent: number;
        winRate: number;
        profitFactor: number;
        totalTrades: number;
        averageWin: number;
        averageLoss: number;
        volatility: number;
        calmarRatio: number;
    };
    trades: any[];
    score: number;
    rank: number;
}
export declare class HyperparameterOptimizer {
    private marketData;
    private baseConfig;
    private optimizationRanges;
    constructor();
    /**
     * Run comprehensive hyperparameter optimization
     */
    runOptimization(numIterations?: number): Promise<OptimizationResult[]>;
    /**
     * Load market data for optimization
     */
    private loadMarketData;
    /**
     * Generate parameter configurations for testing
     */
    private generateParameterConfigurations;
    /**
     * Generate grid search configurations (systematic exploration)
     */
    private generateGridSearchConfigs;
    /**
     * Generate random search configurations
     */
    private generateRandomSearchConfigs;
    /**
     * Test a specific parameter configuration
     */
    private testConfiguration;
    /**
     * Calculate composite optimization score
     */
    private calculateOptimizationScore;
    /**
     * Rank optimization results
     */
    private rankResults;
    private createBaseConfig;
    private defineOptimizationRanges;
    private randomChoice;
}
export declare function createHyperparameterOptimizer(): HyperparameterOptimizer;
