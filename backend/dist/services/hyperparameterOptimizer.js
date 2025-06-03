"use strict";
/**
 * Hyperparameter Optimization Engine
 * Systematically tests different parameter combinations to maximize trading performance
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.HyperparameterOptimizer = void 0;
exports.createHyperparameterOptimizer = createHyperparameterOptimizer;
const marketDataProvider_1 = require("./marketDataProvider");
const technicalAnalysis_1 = require("../utils/technicalAnalysis");
const portfolioManager_1 = require("./portfolioManager");
const performanceAnalytics_1 = require("../utils/performanceAnalytics");
const intelligentTradingSystem_1 = require("./intelligentTradingSystem");
const logger_1 = require("../utils/logger");
class HyperparameterOptimizer {
    constructor() {
        this.marketData = [];
        this.baseConfig = this.createBaseConfig();
        this.optimizationRanges = this.defineOptimizationRanges();
    }
    /**
     * Run comprehensive hyperparameter optimization
     */
    async runOptimization(numIterations = 100) {
        const startTime = Date.now();
        logger_1.logger.info('🔬 Starting Comprehensive Hyperparameter Optimization...', {
            iterations: numIterations,
            targetMetrics: ['Sharpe Ratio', 'Total Return', 'Max Drawdown'],
        });
        // Load market data once for all tests
        await this.loadMarketData();
        // Generate parameter combinations
        const parameterConfigs = this.generateParameterConfigurations(numIterations);
        logger_1.logger.info(`📊 Generated ${parameterConfigs.length} parameter configurations`);
        // Test each configuration
        const results = [];
        for (let i = 0; i < parameterConfigs.length; i++) {
            const config = parameterConfigs[i];
            try {
                const result = await this.testConfiguration(config);
                results.push(result);
                // Progress logging
                if ((i + 1) % 10 === 0) {
                    const progress = ((i + 1) / parameterConfigs.length * 100).toFixed(1);
                    const elapsed = (Date.now() - startTime) / 1000;
                    const eta = (elapsed / (i + 1)) * (parameterConfigs.length - i - 1);
                    logger_1.logger.info(`🔬 Optimization Progress: ${progress}% (${i + 1}/${parameterConfigs.length})`, {
                        elapsed: `${elapsed.toFixed(1)}s`,
                        eta: `${eta.toFixed(1)}s`,
                        bestSharpe: results.length > 0 ? Math.max(...results.map(r => r.performance.sharpeRatio)).toFixed(2) : 'N/A',
                    });
                }
            }
            catch (error) {
                logger_1.logger.warn(`⚠️ Failed to test configuration ${config.id}:`, error);
            }
        }
        // Rank and sort results
        const rankedResults = this.rankResults(results);
        const duration = (Date.now() - startTime) / 1000;
        logger_1.logger.info('✅ Hyperparameter optimization completed', {
            duration: `${duration.toFixed(1)}s`,
            totalConfigurations: parameterConfigs.length,
            successfulTests: results.length,
            bestSharpe: rankedResults[0]?.performance.sharpeRatio.toFixed(2),
            bestReturn: rankedResults[0]?.performance.totalReturnPercent.toFixed(2),
        });
        return rankedResults;
    }
    /**
     * Load market data for optimization
     */
    async loadMarketData() {
        logger_1.logger.info('📊 Loading market data for optimization...');
        const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
            symbol: this.baseConfig.symbol,
            timeframe: this.baseConfig.timeframe,
            startDate: this.baseConfig.startDate,
            endDate: this.baseConfig.endDate,
            exchange: 'enhanced-mock',
        });
        // Enhance with comprehensive technical indicators
        const closes = response.data.map(d => d.close);
        const volumes = response.data.map(d => d.volume);
        const highs = response.data.map(d => d.high);
        const lows = response.data.map(d => d.low);
        const sma20 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 20);
        const sma50 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 50);
        const ema12 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 12);
        const ema26 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 26);
        const rsi = technicalAnalysis_1.technicalAnalysis.calculateRSI(closes, 14);
        const macd = technicalAnalysis_1.technicalAnalysis.calculateMACD(closes, 12, 26, 9);
        const bollinger = technicalAnalysis_1.technicalAnalysis.calculateBollingerBands(closes, 20, 2);
        const volumeSMA = technicalAnalysis_1.technicalAnalysis.calculateSMA(volumes, 20);
        const stochastic = technicalAnalysis_1.technicalAnalysis.calculateStochastic(highs, lows, closes, 14, 3);
        const atr = technicalAnalysis_1.technicalAnalysis.calculateATR(highs, lows, closes, 14);
        this.marketData = response.data.map((point, index) => ({
            ...point,
            indicators: {
                sma_20: sma20[index],
                sma_50: sma50[index],
                ema_12: ema12[index],
                ema_26: ema26[index],
                rsi: rsi[index],
                macd: macd.macd[index],
                macd_signal: macd.signal[index],
                macd_histogram: macd.histogram[index],
                bollinger_upper: bollinger.upper[index],
                bollinger_middle: bollinger.middle[index],
                bollinger_lower: bollinger.lower[index],
                volume_sma: volumeSMA[index],
                stochastic_k: stochastic.k[index],
                stochastic_d: stochastic.d[index],
                atr: atr[index],
            },
        }));
        logger_1.logger.info(`✅ Loaded and enhanced ${this.marketData.length} data points for optimization`);
    }
    /**
     * Generate parameter configurations for testing
     */
    generateParameterConfigurations(numConfigs) {
        const configs = [];
        // Use a combination of grid search and random search
        const gridConfigs = this.generateGridSearchConfigs();
        const randomConfigs = this.generateRandomSearchConfigs(numConfigs - gridConfigs.length);
        configs.push(...gridConfigs);
        configs.push(...randomConfigs);
        return configs.slice(0, numConfigs);
    }
    /**
     * Generate grid search configurations (systematic exploration)
     */
    generateGridSearchConfigs() {
        const configs = [];
        let id = 1;
        // Key parameter combinations to test systematically
        const keyConfidences = [60, 70, 80];
        const keyConsensus = [0.5, 0.6, 0.7];
        const keyCooldowns = [5, 15, 30]; // minutes
        const keyRisks = [1, 2, 3];
        for (const confidence of keyConfidences) {
            for (const consensus of keyConsensus) {
                for (const cooldown of keyCooldowns) {
                    for (const risk of keyRisks) {
                        configs.push({
                            id: `grid_${id++}`,
                            minConfidence: confidence,
                            modelConsensus: consensus,
                            decisionCooldown: cooldown,
                            riskPerTrade: risk,
                            stopLossPercent: 1.5,
                            takeProfitMultiplier: 2.5,
                            positionSizeMultiplier: 0.8,
                            trendThreshold: 0.001,
                            volatilityThreshold: 0.3,
                        });
                    }
                }
            }
        }
        return configs;
    }
    /**
     * Generate random search configurations
     */
    generateRandomSearchConfigs(numConfigs) {
        const configs = [];
        for (let i = 0; i < numConfigs; i++) {
            configs.push({
                id: `random_${i + 1}`,
                minConfidence: this.randomChoice(this.optimizationRanges.minConfidence),
                modelConsensus: this.randomChoice(this.optimizationRanges.modelConsensus),
                decisionCooldown: this.randomChoice(this.optimizationRanges.decisionCooldown),
                riskPerTrade: this.randomChoice(this.optimizationRanges.riskPerTrade),
                stopLossPercent: this.randomChoice(this.optimizationRanges.stopLossPercent),
                takeProfitMultiplier: this.randomChoice(this.optimizationRanges.takeProfitMultiplier),
                positionSizeMultiplier: this.randomChoice(this.optimizationRanges.positionSizeMultiplier),
                trendThreshold: this.randomChoice(this.optimizationRanges.trendThreshold),
                volatilityThreshold: this.randomChoice(this.optimizationRanges.volatilityThreshold),
            });
        }
        return configs;
    }
    /**
     * Test a specific parameter configuration
     */
    async testConfiguration(paramConfig) {
        // Create optimized trading system with custom parameters
        const strategy = new OptimizedIntelligentTradingSystem(paramConfig);
        // Create test configuration
        const testConfig = {
            ...this.baseConfig,
            riskPerTrade: paramConfig.riskPerTrade,
        };
        strategy.initialize(testConfig);
        const portfolioManager = new portfolioManager_1.PortfolioManager(testConfig);
        let signalCount = 0;
        let tradeCount = 0;
        // Run backtest with optimized parameters
        for (let i = 0; i < this.marketData.length; i++) {
            const currentCandle = this.marketData[i];
            portfolioManager.updatePositions(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            const closedTrades = portfolioManager.checkStopLossAndTakeProfit(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            tradeCount += closedTrades.length;
            const signal = strategy.generateSignal(this.marketData, i);
            if (signal && signal.confidence > 0) {
                signalCount++;
                const trade = portfolioManager.executeTrade(signal, currentCandle.close, currentCandle.timestamp);
                if (trade) {
                    trade.strategy = `Optimized_${paramConfig.id}`;
                    tradeCount++;
                }
            }
            if (i % 24 === 0 || signal) {
                portfolioManager.createSnapshot(currentCandle.timestamp);
            }
        }
        // Calculate performance
        const trades = portfolioManager.getTrades();
        const portfolioHistory = portfolioManager.getPortfolioHistory();
        const performance = performanceAnalytics_1.PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, testConfig);
        // Calculate composite optimization score
        const score = this.calculateOptimizationScore(performance);
        return {
            config: paramConfig,
            performance: {
                totalReturnPercent: performance.totalReturnPercent,
                sharpeRatio: performance.sharpeRatio,
                maxDrawdownPercent: performance.maxDrawdownPercent,
                winRate: performance.winRate,
                profitFactor: performance.profitFactor,
                totalTrades: performance.totalTrades,
                averageWin: performance.averageWin,
                averageLoss: performance.averageLoss,
                volatility: performance.volatility,
                calmarRatio: performance.calmarRatio,
            },
            trades,
            score,
            rank: 0, // Will be set during ranking
        };
    }
    /**
     * Calculate composite optimization score
     */
    calculateOptimizationScore(performance) {
        // Weighted scoring system
        const sharpeWeight = 0.4;
        const returnWeight = 0.3;
        const drawdownWeight = 0.2;
        const tradeCountWeight = 0.1;
        // Normalize metrics (higher is better)
        const sharpeScore = Math.max(0, Math.min(100, (performance.sharpeRatio + 2) * 25)); // -2 to 2 -> 0 to 100
        const returnScore = Math.max(0, Math.min(100, performance.totalReturnPercent + 50)); // -50% to 50% -> 0 to 100
        const drawdownScore = Math.max(0, 100 - performance.maxDrawdownPercent * 2); // 0% to 50% -> 100 to 0
        const tradeScore = Math.min(100, performance.totalTrades * 5); // More trades up to 20 -> 100
        const compositeScore = sharpeScore * sharpeWeight +
            returnScore * returnWeight +
            drawdownScore * drawdownWeight +
            tradeScore * tradeCountWeight;
        return compositeScore;
    }
    /**
     * Rank optimization results
     */
    rankResults(results) {
        // Sort by composite score (descending)
        const sorted = results.sort((a, b) => b.score - a.score);
        // Assign ranks
        sorted.forEach((result, index) => {
            result.rank = index + 1;
        });
        return sorted;
    }
    // Helper methods
    createBaseConfig() {
        const endDate = new Date();
        const startDate = new Date(endDate.getTime() - (30 * 24 * 60 * 60 * 1000));
        return {
            symbol: 'BTCUSD',
            timeframe: '1h',
            startDate,
            endDate,
            initialCapital: 2000,
            leverage: 3,
            riskPerTrade: 2, // Will be overridden during optimization
            commission: 0.1,
            slippage: 0.05,
            strategy: 'Optimized_Intelligent_AI',
            parameters: {},
        };
    }
    defineOptimizationRanges() {
        return {
            minConfidence: [50, 55, 60, 65, 70, 75, 80, 85, 90],
            modelConsensus: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            decisionCooldown: [1, 3, 5, 10, 15, 20, 25, 30], // minutes
            riskPerTrade: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            stopLossPercent: [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
            takeProfitMultiplier: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            positionSizeMultiplier: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            trendThreshold: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003],
            volatilityThreshold: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        };
    }
    randomChoice(array) {
        return array[Math.floor(Math.random() * array.length)];
    }
}
exports.HyperparameterOptimizer = HyperparameterOptimizer;
/**
 * Optimized version of IntelligentTradingSystem with configurable parameters
 */
class OptimizedIntelligentTradingSystem extends intelligentTradingSystem_1.IntelligentTradingSystem {
    constructor(config) {
        super();
        this.optimizationConfig = config;
        // Override parameters with optimization config
        this.parameters = {
            ...this.parameters,
            minConfidence: config.minConfidence,
            minModelConsensus: config.modelConsensus,
            decisionCooldown: config.decisionCooldown * 60 * 1000, // Convert to milliseconds
            stopLossPercent: config.stopLossPercent,
            takeProfitMultiplier: config.takeProfitMultiplier,
            positionSizeMultiplier: config.positionSizeMultiplier,
            trendThreshold: config.trendThreshold,
            volatilityThreshold: config.volatilityThreshold,
        };
    }
    // Override methods to use optimization parameters
    getMinConfidence() {
        return this.optimizationConfig.minConfidence;
    }
    getModelConsensusThreshold() {
        return this.optimizationConfig.modelConsensus;
    }
    getDecisionCooldown() {
        return this.optimizationConfig.decisionCooldown * 60 * 1000;
    }
}
// Export factory function
function createHyperparameterOptimizer() {
    return new HyperparameterOptimizer();
}
//# sourceMappingURL=hyperparameterOptimizer.js.map