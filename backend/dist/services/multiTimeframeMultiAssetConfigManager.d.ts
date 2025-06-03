/**
 * Multi-Timeframe Multi-Asset Configuration Manager
 * Manages different backtesting configurations and scenarios
 */
import { MultiTimeframeMultiAssetBacktestConfig, TimeframeAssetConfig } from './multiTimeframeMultiAssetBacktester';
import { Timeframe } from './multiTimeframeDataProvider';
export type BacktestScenario = 'CONSERVATIVE_SINGLE_ASSET' | 'AGGRESSIVE_SINGLE_ASSET' | 'CONSERVATIVE_MULTI_ASSET' | 'AGGRESSIVE_MULTI_ASSET' | 'BALANCED_PORTFOLIO' | 'HIGH_FREQUENCY_SCALPING' | 'SWING_TRADING' | 'TREND_FOLLOWING' | 'MEAN_REVERSION' | 'CORRELATION_ARBITRAGE';
export type TimeframeStrategy = 'SHORT_TERM_FOCUS' | 'MEDIUM_TERM_FOCUS' | 'LONG_TERM_FOCUS' | 'HIERARCHICAL_BALANCED' | 'TREND_CONFIRMATION' | 'SCALPING_OPTIMIZED';
export interface ScenarioConfig {
    scenario: BacktestScenario;
    description: string;
    assetConfigs: TimeframeAssetConfig[];
    primaryTimeframe: Timeframe;
    timeframeWeights: {
        [timeframe in Timeframe]?: number;
    };
    portfolioMode: 'SINGLE_ASSET' | 'MULTI_ASSET' | 'DYNAMIC';
    rebalanceFrequency: 'SIGNAL_BASED' | 'HOURLY' | 'DAILY' | 'WEEKLY';
    riskProfile: 'CONSERVATIVE' | 'MODERATE' | 'AGGRESSIVE';
    expectedDuration: number;
    recommendedCapital: number;
}
export declare class MultiTimeframeMultiAssetConfigManager {
    private predefinedScenarios;
    constructor();
    /**
     * Initialize predefined backtesting scenarios
     */
    private initializePredefinedScenarios;
    /**
     * Get configuration for a specific scenario
     */
    getScenarioConfig(scenario: BacktestScenario): ScenarioConfig | null;
    /**
     * Create full backtest configuration from scenario
     */
    createBacktestConfig(scenario: BacktestScenario, startDate: Date, endDate: Date, customParams?: Partial<MultiTimeframeMultiAssetBacktestConfig>): MultiTimeframeMultiAssetBacktestConfig | null;
    /**
     * Get all available scenarios
     */
    getAvailableScenarios(): BacktestScenario[];
    /**
     * Get scenarios by risk profile
     */
    getScenariosByRiskProfile(riskProfile: 'CONSERVATIVE' | 'MODERATE' | 'AGGRESSIVE'): BacktestScenario[];
    /**
     * Get scenarios by portfolio mode
     */
    getScenariosByPortfolioMode(portfolioMode: 'SINGLE_ASSET' | 'MULTI_ASSET' | 'DYNAMIC'): BacktestScenario[];
    /**
     * Create custom scenario configuration
     */
    createCustomScenario(name: string, description: string, assetConfigs: TimeframeAssetConfig[], options?: {
        primaryTimeframe?: Timeframe;
        timeframeWeights?: {
            [timeframe in Timeframe]?: number;
        };
        portfolioMode?: 'SINGLE_ASSET' | 'MULTI_ASSET' | 'DYNAMIC';
        rebalanceFrequency?: 'SIGNAL_BASED' | 'HOURLY' | 'DAILY' | 'WEEKLY';
        riskProfile?: 'CONSERVATIVE' | 'MODERATE' | 'AGGRESSIVE';
        recommendedCapital?: number;
    }): ScenarioConfig;
    /**
     * Validate scenario configuration
     */
    validateScenarioConfig(config: ScenarioConfig): {
        valid: boolean;
        errors: string[];
    };
    private getLeverageForRiskProfile;
    private getRiskPerTradeForProfile;
    /**
     * Get scenario recommendations based on market conditions
     */
    getScenarioRecommendations(marketCondition: 'BULL' | 'BEAR' | 'SIDEWAYS' | 'VOLATILE'): BacktestScenario[];
}
export declare function createMultiTimeframeMultiAssetConfigManager(): MultiTimeframeMultiAssetConfigManager;
