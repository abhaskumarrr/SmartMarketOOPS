"use strict";
/**
 * Multi-Timeframe Multi-Asset Configuration Manager
 * Manages different backtesting configurations and scenarios
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeMultiAssetConfigManager = void 0;
exports.createMultiTimeframeMultiAssetConfigManager = createMultiTimeframeMultiAssetConfigManager;
const logger_1 = require("../utils/logger");
class MultiTimeframeMultiAssetConfigManager {
    constructor() {
        this.predefinedScenarios = new Map();
        this.initializePredefinedScenarios();
        logger_1.logger.info('🔧 Multi-Timeframe Multi-Asset Config Manager initialized', {
            scenarios: Array.from(this.predefinedScenarios.keys()),
        });
    }
    /**
     * Initialize predefined backtesting scenarios
     */
    initializePredefinedScenarios() {
        // Conservative Single Asset Scenarios
        this.predefinedScenarios.set('CONSERVATIVE_SINGLE_ASSET', {
            scenario: 'CONSERVATIVE_SINGLE_ASSET',
            description: 'Conservative single-asset strategy focusing on Bitcoin with multiple timeframe confirmation',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['1d', '4h', '1h'],
                    priority: 'PRIMARY',
                    weight: 1.0,
                },
            ],
            primaryTimeframe: '1h',
            timeframeWeights: { '1d': 0.5, '4h': 0.3, '1h': 0.2 },
            portfolioMode: 'SINGLE_ASSET',
            rebalanceFrequency: 'DAILY',
            riskProfile: 'CONSERVATIVE',
            expectedDuration: 24,
            recommendedCapital: 10000,
        });
        this.predefinedScenarios.set('AGGRESSIVE_SINGLE_ASSET', {
            scenario: 'AGGRESSIVE_SINGLE_ASSET',
            description: 'Aggressive single-asset strategy with high-frequency signals across multiple timeframes',
            assetConfigs: [
                {
                    asset: 'SOLUSD',
                    timeframes: ['1h', '15m', '5m', '1m'],
                    priority: 'PRIMARY',
                    weight: 1.0,
                },
            ],
            primaryTimeframe: '15m',
            timeframeWeights: { '1h': 0.4, '15m': 0.3, '5m': 0.2, '1m': 0.1 },
            portfolioMode: 'SINGLE_ASSET',
            rebalanceFrequency: 'SIGNAL_BASED',
            riskProfile: 'AGGRESSIVE',
            expectedDuration: 12,
            recommendedCapital: 25000,
        });
        // Multi-Asset Scenarios
        this.predefinedScenarios.set('CONSERVATIVE_MULTI_ASSET', {
            scenario: 'CONSERVATIVE_MULTI_ASSET',
            description: 'Conservative multi-asset portfolio with equal allocation and long-term focus',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['1d', '4h', '1h'],
                    priority: 'PRIMARY',
                    weight: 0.4,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['1d', '4h', '1h'],
                    priority: 'PRIMARY',
                    weight: 0.35,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['4h', '1h'],
                    priority: 'SECONDARY',
                    weight: 0.25,
                },
            ],
            primaryTimeframe: '1h',
            timeframeWeights: { '1d': 0.4, '4h': 0.35, '1h': 0.25 },
            portfolioMode: 'MULTI_ASSET',
            rebalanceFrequency: 'DAILY',
            riskProfile: 'CONSERVATIVE',
            expectedDuration: 48,
            recommendedCapital: 50000,
        });
        this.predefinedScenarios.set('AGGRESSIVE_MULTI_ASSET', {
            scenario: 'AGGRESSIVE_MULTI_ASSET',
            description: 'Aggressive multi-asset strategy with dynamic allocation and frequent rebalancing',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['4h', '1h', '15m', '5m'],
                    priority: 'PRIMARY',
                    weight: 0.35,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['4h', '1h', '15m', '5m'],
                    priority: 'PRIMARY',
                    weight: 0.35,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['1h', '15m', '5m', '1m'],
                    priority: 'PRIMARY',
                    weight: 0.3,
                },
            ],
            primaryTimeframe: '15m',
            timeframeWeights: { '4h': 0.3, '1h': 0.3, '15m': 0.25, '5m': 0.1, '1m': 0.05 },
            portfolioMode: 'DYNAMIC',
            rebalanceFrequency: 'HOURLY',
            riskProfile: 'AGGRESSIVE',
            expectedDuration: 24,
            recommendedCapital: 100000,
        });
        this.predefinedScenarios.set('BALANCED_PORTFOLIO', {
            scenario: 'BALANCED_PORTFOLIO',
            description: 'Balanced portfolio approach with moderate risk and diversified timeframe analysis',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['1d', '4h', '1h', '15m'],
                    priority: 'PRIMARY',
                    weight: 0.4,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['4h', '1h', '15m'],
                    priority: 'PRIMARY',
                    weight: 0.35,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['1h', '15m', '5m'],
                    priority: 'SECONDARY',
                    weight: 0.25,
                },
            ],
            primaryTimeframe: '1h',
            timeframeWeights: { '1d': 0.25, '4h': 0.25, '1h': 0.25, '15m': 0.15, '5m': 0.1 },
            portfolioMode: 'MULTI_ASSET',
            rebalanceFrequency: 'DAILY',
            riskProfile: 'MODERATE',
            expectedDuration: 36,
            recommendedCapital: 75000,
        });
        // Specialized Strategy Scenarios
        this.predefinedScenarios.set('HIGH_FREQUENCY_SCALPING', {
            scenario: 'HIGH_FREQUENCY_SCALPING',
            description: 'High-frequency scalping strategy focusing on short timeframes with quick entries and exits',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['15m', '5m', '1m'],
                    priority: 'PRIMARY',
                    weight: 0.5,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['15m', '5m', '1m'],
                    priority: 'PRIMARY',
                    weight: 0.5,
                },
            ],
            primaryTimeframe: '5m',
            timeframeWeights: { '15m': 0.3, '5m': 0.5, '1m': 0.2 },
            portfolioMode: 'DYNAMIC',
            rebalanceFrequency: 'SIGNAL_BASED',
            riskProfile: 'AGGRESSIVE',
            expectedDuration: 8,
            recommendedCapital: 50000,
        });
        this.predefinedScenarios.set('SWING_TRADING', {
            scenario: 'SWING_TRADING',
            description: 'Swing trading strategy focusing on medium-term trends with position holding',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['1d', '4h', '1h'],
                    priority: 'PRIMARY',
                    weight: 0.4,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['1d', '4h', '1h'],
                    priority: 'PRIMARY',
                    weight: 0.35,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['4h', '1h'],
                    priority: 'SECONDARY',
                    weight: 0.25,
                },
            ],
            primaryTimeframe: '4h',
            timeframeWeights: { '1d': 0.4, '4h': 0.4, '1h': 0.2 },
            portfolioMode: 'MULTI_ASSET',
            rebalanceFrequency: 'WEEKLY',
            riskProfile: 'MODERATE',
            expectedDuration: 72,
            recommendedCapital: 30000,
        });
        this.predefinedScenarios.set('TREND_FOLLOWING', {
            scenario: 'TREND_FOLLOWING',
            description: 'Trend following strategy using higher timeframes for direction and lower for entry',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['1d', '4h', '1h', '15m'],
                    priority: 'PRIMARY',
                    weight: 0.5,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['1d', '4h', '1h'],
                    priority: 'PRIMARY',
                    weight: 0.3,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['4h', '1h'],
                    priority: 'SECONDARY',
                    weight: 0.2,
                },
            ],
            primaryTimeframe: '1h',
            timeframeWeights: { '1d': 0.4, '4h': 0.3, '1h': 0.2, '15m': 0.1 },
            portfolioMode: 'MULTI_ASSET',
            rebalanceFrequency: 'DAILY',
            riskProfile: 'MODERATE',
            expectedDuration: 48,
            recommendedCapital: 40000,
        });
        this.predefinedScenarios.set('MEAN_REVERSION', {
            scenario: 'MEAN_REVERSION',
            description: 'Mean reversion strategy identifying oversold/overbought conditions across timeframes',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['4h', '1h', '15m'],
                    priority: 'PRIMARY',
                    weight: 0.4,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['4h', '1h', '15m'],
                    priority: 'PRIMARY',
                    weight: 0.35,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['1h', '15m', '5m'],
                    priority: 'PRIMARY',
                    weight: 0.25,
                },
            ],
            primaryTimeframe: '1h',
            timeframeWeights: { '4h': 0.3, '1h': 0.4, '15m': 0.2, '5m': 0.1 },
            portfolioMode: 'MULTI_ASSET',
            rebalanceFrequency: 'HOURLY',
            riskProfile: 'MODERATE',
            expectedDuration: 24,
            recommendedCapital: 60000,
        });
        this.predefinedScenarios.set('CORRELATION_ARBITRAGE', {
            scenario: 'CORRELATION_ARBITRAGE',
            description: 'Correlation arbitrage strategy exploiting price differences between correlated assets',
            assetConfigs: [
                {
                    asset: 'BTCUSD',
                    timeframes: ['1h', '15m', '5m'],
                    priority: 'PRIMARY',
                    weight: 0.4,
                },
                {
                    asset: 'ETHUSD',
                    timeframes: ['1h', '15m', '5m'],
                    priority: 'PRIMARY',
                    weight: 0.4,
                },
                {
                    asset: 'SOLUSD',
                    timeframes: ['1h', '15m', '5m'],
                    priority: 'SECONDARY',
                    weight: 0.2,
                },
            ],
            primaryTimeframe: '15m',
            timeframeWeights: { '1h': 0.4, '15m': 0.4, '5m': 0.2 },
            portfolioMode: 'DYNAMIC',
            rebalanceFrequency: 'SIGNAL_BASED',
            riskProfile: 'AGGRESSIVE',
            expectedDuration: 16,
            recommendedCapital: 80000,
        });
    }
    /**
     * Get configuration for a specific scenario
     */
    getScenarioConfig(scenario) {
        return this.predefinedScenarios.get(scenario) || null;
    }
    /**
     * Create full backtest configuration from scenario
     */
    createBacktestConfig(scenario, startDate, endDate, customParams) {
        const scenarioConfig = this.getScenarioConfig(scenario);
        if (!scenarioConfig) {
            logger_1.logger.error(`❌ Unknown scenario: ${scenario}`);
            return null;
        }
        const baseConfig = {
            symbol: 'PORTFOLIO',
            timeframe: scenarioConfig.primaryTimeframe,
            startDate,
            endDate,
            initialCapital: scenarioConfig.recommendedCapital,
            leverage: this.getLeverageForRiskProfile(scenarioConfig.riskProfile),
            riskPerTrade: this.getRiskPerTradeForProfile(scenarioConfig.riskProfile),
            commission: 0.1,
            slippage: 0.05,
            strategy: scenario,
            parameters: {},
            // Multi-timeframe multi-asset specific
            assetConfigs: scenarioConfig.assetConfigs,
            primaryTimeframe: scenarioConfig.primaryTimeframe,
            timeframeWeights: scenarioConfig.timeframeWeights,
            portfolioMode: scenarioConfig.portfolioMode,
            rebalanceFrequency: scenarioConfig.rebalanceFrequency,
        };
        // Apply custom parameters if provided
        if (customParams) {
            Object.assign(baseConfig, customParams);
        }
        logger_1.logger.info(`🔧 Created backtest config for ${scenario}`, {
            assets: baseConfig.assetConfigs.map(c => c.asset),
            timeframes: baseConfig.assetConfigs.flatMap(c => c.timeframes),
            primaryTimeframe: baseConfig.primaryTimeframe,
            portfolioMode: baseConfig.portfolioMode,
            capital: baseConfig.initialCapital,
        });
        return baseConfig;
    }
    /**
     * Get all available scenarios
     */
    getAvailableScenarios() {
        return Array.from(this.predefinedScenarios.keys());
    }
    /**
     * Get scenarios by risk profile
     */
    getScenariosByRiskProfile(riskProfile) {
        return Array.from(this.predefinedScenarios.entries())
            .filter(([_, config]) => config.riskProfile === riskProfile)
            .map(([scenario, _]) => scenario);
    }
    /**
     * Get scenarios by portfolio mode
     */
    getScenariosByPortfolioMode(portfolioMode) {
        return Array.from(this.predefinedScenarios.entries())
            .filter(([_, config]) => config.portfolioMode === portfolioMode)
            .map(([scenario, _]) => scenario);
    }
    /**
     * Create custom scenario configuration
     */
    createCustomScenario(name, description, assetConfigs, options = {}) {
        const customScenario = {
            scenario: 'BALANCED_PORTFOLIO', // Default scenario type
            description,
            assetConfigs,
            primaryTimeframe: options.primaryTimeframe || '1h',
            timeframeWeights: options.timeframeWeights || { '1h': 1.0 },
            portfolioMode: options.portfolioMode || 'MULTI_ASSET',
            rebalanceFrequency: options.rebalanceFrequency || 'DAILY',
            riskProfile: options.riskProfile || 'MODERATE',
            expectedDuration: 24,
            recommendedCapital: options.recommendedCapital || 50000,
        };
        logger_1.logger.info(`🔧 Created custom scenario: ${name}`, {
            assets: assetConfigs.map(c => c.asset),
            timeframes: assetConfigs.flatMap(c => c.timeframes),
            portfolioMode: customScenario.portfolioMode,
        });
        return customScenario;
    }
    /**
     * Validate scenario configuration
     */
    validateScenarioConfig(config) {
        const errors = [];
        // Validate asset configs
        if (!config.assetConfigs || config.assetConfigs.length === 0) {
            errors.push('At least one asset configuration is required');
        }
        // Validate timeframe weights
        if (config.timeframeWeights) {
            const totalWeight = Object.values(config.timeframeWeights).reduce((sum, weight) => sum + (weight || 0), 0);
            if (Math.abs(totalWeight - 1.0) > 0.01) {
                errors.push(`Timeframe weights must sum to 1.0, got ${totalWeight.toFixed(3)}`);
            }
        }
        // Validate asset weights
        const totalAssetWeight = config.assetConfigs.reduce((sum, asset) => sum + asset.weight, 0);
        if (Math.abs(totalAssetWeight - 1.0) > 0.01) {
            errors.push(`Asset weights must sum to 1.0, got ${totalAssetWeight.toFixed(3)}`);
        }
        // Validate primary timeframe is included
        const allTimeframes = config.assetConfigs.flatMap(c => c.timeframes);
        if (!allTimeframes.includes(config.primaryTimeframe)) {
            errors.push(`Primary timeframe ${config.primaryTimeframe} must be included in asset configurations`);
        }
        return {
            valid: errors.length === 0,
            errors,
        };
    }
    // Helper methods
    getLeverageForRiskProfile(riskProfile) {
        switch (riskProfile) {
            case 'CONSERVATIVE': return 1;
            case 'MODERATE': return 2;
            case 'AGGRESSIVE': return 3;
            default: return 1;
        }
    }
    getRiskPerTradeForProfile(riskProfile) {
        switch (riskProfile) {
            case 'CONSERVATIVE': return 1;
            case 'MODERATE': return 2;
            case 'AGGRESSIVE': return 3;
            default: return 2;
        }
    }
    /**
     * Get scenario recommendations based on market conditions
     */
    getScenarioRecommendations(marketCondition) {
        switch (marketCondition) {
            case 'BULL':
                return ['TREND_FOLLOWING', 'AGGRESSIVE_MULTI_ASSET', 'SWING_TRADING'];
            case 'BEAR':
                return ['MEAN_REVERSION', 'CONSERVATIVE_MULTI_ASSET', 'HIGH_FREQUENCY_SCALPING'];
            case 'SIDEWAYS':
                return ['MEAN_REVERSION', 'CORRELATION_ARBITRAGE', 'BALANCED_PORTFOLIO'];
            case 'VOLATILE':
                return ['HIGH_FREQUENCY_SCALPING', 'AGGRESSIVE_SINGLE_ASSET', 'CORRELATION_ARBITRAGE'];
            default:
                return ['BALANCED_PORTFOLIO', 'CONSERVATIVE_MULTI_ASSET'];
        }
    }
}
exports.MultiTimeframeMultiAssetConfigManager = MultiTimeframeMultiAssetConfigManager;
// Export factory function
function createMultiTimeframeMultiAssetConfigManager() {
    return new MultiTimeframeMultiAssetConfigManager();
}
//# sourceMappingURL=multiTimeframeMultiAssetConfigManager.js.map