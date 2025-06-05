"use strict";
/**
 * Intelligent Trading Bot Backtester
 * Comprehensive backtesting with $10 capital, 200x leverage on ETH using 1 year data
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntelligentTradingBotBacktester = void 0;
const MultiTimeframeAnalysisEngine_1 = require("../services/MultiTimeframeAnalysisEngine");
const EnhancedMarketRegimeDetector_1 = require("../services/EnhancedMarketRegimeDetector");
const AdaptiveStopLossSystem_1 = require("../services/AdaptiveStopLossSystem");
const SmartTakeProfitSystem_1 = require("../services/SmartTakeProfitSystem");
const EnhancedMLIntegrationService_1 = require("../services/EnhancedMLIntegrationService");
const AdvancedSignalFilteringSystem_1 = require("../services/AdvancedSignalFilteringSystem");
const logger_1 = require("../utils/logger");
class IntelligentTradingBotBacktester {
    constructor(config) {
        this.totalTrades = 0;
        this.positions = [];
        this.activePositions = new Map();
        this.dailyBalances = [];
        this.config = config;
        this.currentBalance = config.initialCapital;
        this.peakBalance = config.initialCapital;
        this.currentDrawdown = 0;
        this.maxDrawdown = 0;
        // Initialize services with mock data service
        const mockDataService = this.createMockDataService();
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine(mockDataService);
        this.regimeDetector = new EnhancedMarketRegimeDetector_1.EnhancedMarketRegimeDetector(mockDataService);
        this.stopLossSystem = new AdaptiveStopLossSystem_1.AdaptiveStopLossSystem(mockDataService);
        this.takeProfitSystem = new SmartTakeProfitSystem_1.SmartTakeProfitSystem(mockDataService);
        this.mlService = new EnhancedMLIntegrationService_1.EnhancedMLIntegrationService(mockDataService);
        this.signalFilter = new AdvancedSignalFilteringSystem_1.AdvancedSignalFilteringSystem(mockDataService);
    }
    /**
     * Run comprehensive backtest
     */
    async runBacktest() {
        logger_1.logger.info('🚀 Starting Intelligent Trading Bot Backtest');
        logger_1.logger.info(`💰 Capital: $${this.config.initialCapital}, Leverage: ${this.config.leverage}x`);
        logger_1.logger.info(`📊 Symbol: ${this.config.symbol}, Period: ${this.config.startDate} to ${this.config.endDate}`);
        // Generate historical data for ETH
        const historicalData = await this.generateETHHistoricalData();
        // Process each day
        for (let i = 0; i < historicalData.length; i++) {
            const currentData = historicalData[i];
            const date = new Date(currentData.timestamp).toISOString().split('T')[0];
            // Update active positions
            await this.updateActivePositions(currentData);
            // Look for new opportunities
            if (this.activePositions.size < this.config.maxPositions) {
                await this.evaluateNewOpportunity(currentData, historicalData.slice(Math.max(0, i - 100), i + 1));
            }
            // Update daily balance tracking
            this.updateDailyTracking(date);
            // Log progress every 30 days
            if (i % 30 === 0) {
                logger_1.logger.info(`📅 Progress: ${date}, Balance: $${this.currentBalance.toFixed(2)}, Trades: ${this.positions.length}`);
            }
        }
        // Close any remaining positions
        await this.closeAllPositions(historicalData[historicalData.length - 1]);
        // Generate comprehensive results
        const results = this.generateBacktestResults();
        this.logBacktestSummary(results);
        return results;
    }
    /**
     * Generate ETH historical data for 1 year
     */
    async generateETHHistoricalData() {
        const data = [];
        const startDate = new Date(this.config.startDate);
        const endDate = new Date(this.config.endDate);
        // ETH price data simulation based on realistic 2023-2024 movements
        let currentPrice = 1800; // Starting price around $1800
        const volatility = 0.04; // 4% daily volatility
        for (let date = new Date(startDate); date <= endDate; date.setDate(date.getDate() + 1)) {
            // Simulate realistic ETH price movements
            const randomFactor = (Math.random() - 0.5) * 2; // -1 to 1
            const trendFactor = this.getETHTrendFactor(date);
            const dailyChange = (randomFactor * volatility + trendFactor) * currentPrice;
            currentPrice = Math.max(800, currentPrice + dailyChange); // Floor at $800
            // Generate OHLCV data
            const high = currentPrice * (1 + Math.random() * 0.03);
            const low = currentPrice * (1 - Math.random() * 0.03);
            const open = currentPrice * (0.98 + Math.random() * 0.04);
            const close = currentPrice;
            const volume = 1000000 + Math.random() * 5000000;
            data.push({
                timestamp: date.getTime(),
                date: date.toISOString().split('T')[0],
                open,
                high,
                low,
                close,
                volume,
                price: close
            });
        }
        logger_1.logger.info(`📈 Generated ${data.length} days of ETH historical data`);
        logger_1.logger.info(`💹 Price range: $${Math.min(...data.map(d => d.low)).toFixed(2)} - $${Math.max(...data.map(d => d.high)).toFixed(2)}`);
        return data;
    }
    /**
     * Get ETH trend factor based on historical patterns
     */
    getETHTrendFactor(date) {
        const month = date.getMonth();
        const dayOfYear = Math.floor((date.getTime() - new Date(date.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24));
        // Simulate seasonal patterns and major events
        let trendFactor = 0;
        // Bull run simulation (Q1 2024)
        if (month >= 0 && month <= 2) {
            trendFactor = 0.002; // Slight upward bias
        }
        // Correction period (Q2)
        else if (month >= 3 && month <= 5) {
            trendFactor = -0.001; // Slight downward bias
        }
        // Summer consolidation (Q3)
        else if (month >= 6 && month <= 8) {
            trendFactor = 0; // Neutral
        }
        // Year-end rally (Q4)
        else {
            trendFactor = 0.0015; // Moderate upward bias
        }
        // Add some noise
        trendFactor += (Math.random() - 0.5) * 0.001;
        return trendFactor;
    }
    /**
     * Evaluate new trading opportunity with advanced signal filtering
     */
    async evaluateNewOpportunity(currentData, historicalWindow) {
        try {
            logger_1.logger.debug(`🔍 Evaluating opportunity at $${currentData.price.toFixed(2)}`);
            // Use advanced signal filtering system for high-quality signals
            const filteredSignal = await this.signalFilter.generateFilteredSignal(this.config.symbol);
            if (filteredSignal && filteredSignal.signal_score >= 85) {
                logger_1.logger.info(`🎯 HIGH-QUALITY SIGNAL: Score ${filteredSignal.signal_score}/100, ML Confidence: ${(filteredSignal.ml_confidence * 100).toFixed(1)}%`);
                // Execute trade with filtered signal parameters
                await this.executeAdvancedFilteredTrade(currentData, filteredSignal);
            }
            else if (filteredSignal) {
                logger_1.logger.debug(`⚠️ Signal quality insufficient: Score ${filteredSignal.signal_score}/100 (need 85+)`);
            }
            else {
                logger_1.logger.debug(`❌ No signal generated - filters rejected opportunity`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error evaluating opportunity:', error);
        }
    }
    /**
     * Execute trade with advanced filtered signal
     */
    async executeAdvancedFilteredTrade(currentData, signal) {
        const positionId = `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        // Dynamic risk management: Start aggressive, become conservative as balance grows
        const dynamicRisk = this.calculateDynamicRisk();
        const dynamicLeverage = this.calculateDynamicLeverage();
        // Apply signal confidence multiplier to position sizing
        const confidenceMultiplier = signal.position_size_multiplier;
        const adjustedRiskPercent = dynamicRisk.riskPercent * confidenceMultiplier;
        // Calculate position size with confidence-adjusted parameters
        const riskAmount = this.currentBalance * (adjustedRiskPercent / 100);
        const notionalValue = riskAmount * dynamicLeverage.leverage;
        const contractSize = notionalValue / currentData.price;
        // Use signal's optimized stop loss and take profit levels
        const stopLossPrice = signal.stop_loss;
        const takeProfitLevels = signal.take_profit_levels;
        const position = {
            id: positionId,
            symbol: this.config.symbol,
            side: signal.side,
            entryPrice: currentData.price,
            size: contractSize,
            leverage: dynamicLeverage.leverage,
            entryTime: currentData.timestamp,
            stopLoss: stopLossPrice,
            takeProfitLevels,
            pnl: 0,
            pnlPercent: 0,
            exitReason: '',
            healthScore: signal.signal_score,
            regimeAtEntry: 'filtered_signal',
            signals: signal
        };
        this.activePositions.set(positionId, position);
        this.totalTrades++;
        logger_1.logger.info(`📈 FILTERED TRADE: ${signal.side} ${contractSize.toFixed(4)} ETH at $${currentData.price.toFixed(2)}`);
        logger_1.logger.info(`🎯 Signal Score: ${signal.signal_score}/100, ML: ${(signal.ml_confidence * 100).toFixed(1)}%, Risk: ${adjustedRiskPercent.toFixed(1)}%`);
        logger_1.logger.info(`🛡️ Stop: $${stopLossPrice.toFixed(2)}, TPs: [${takeProfitLevels.map(tp => `$${tp.toFixed(2)}`).join(', ')}]`);
    }
    /**
     * Execute backtest trade with dynamic risk ladder strategy (legacy method)
     */
    async executeBacktestTrade(currentData, opportunity) {
        const positionId = `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        // Dynamic risk management: Start aggressive, become conservative as balance grows
        const dynamicRisk = this.calculateDynamicRisk();
        const dynamicLeverage = this.calculateDynamicLeverage();
        // Calculate position size with dynamic parameters
        const riskAmount = this.currentBalance * (dynamicRisk.riskPercent / 100);
        const notionalValue = riskAmount * dynamicLeverage.leverage;
        const contractSize = notionalValue / currentData.price;
        // Dynamic stop loss and take profit based on risk level
        const stopLossDistance = dynamicRisk.stopLossPercent / 100;
        const stopLossPrice = opportunity.side === 'LONG'
            ? currentData.price * (1 - stopLossDistance)
            : currentData.price * (1 + stopLossDistance);
        // Dynamic take profit levels - more aggressive when balance is low
        const takeProfitLevels = [
            currentData.price * (opportunity.side === 'LONG' ? (1 + dynamicRisk.takeProfitLevels[0]) : (1 - dynamicRisk.takeProfitLevels[0])),
            currentData.price * (opportunity.side === 'LONG' ? (1 + dynamicRisk.takeProfitLevels[1]) : (1 - dynamicRisk.takeProfitLevels[1])),
            currentData.price * (opportunity.side === 'LONG' ? (1 + dynamicRisk.takeProfitLevels[2]) : (1 - dynamicRisk.takeProfitLevels[2]))
        ];
        const position = {
            id: positionId,
            symbol: this.config.symbol,
            side: opportunity.side,
            entryPrice: currentData.price,
            size: contractSize,
            leverage: dynamicLeverage.leverage,
            entryTime: currentData.timestamp,
            stopLoss: stopLossPrice,
            takeProfitLevels,
            pnl: 0,
            pnlPercent: 0,
            exitReason: '',
            healthScore: opportunity.score,
            regimeAtEntry: opportunity.regime,
            signals: opportunity.signals
        };
        this.activePositions.set(positionId, position);
        this.totalTrades++;
        logger_1.logger.debug(`📈 Opened ${opportunity.side} position: ${contractSize.toFixed(4)} ETH at $${currentData.price.toFixed(2)}`);
        logger_1.logger.debug(`🎯 Score: ${opportunity.score}/100, Risk: ${dynamicRisk.riskPercent}%, Leverage: ${dynamicLeverage.leverage}x`);
        logger_1.logger.debug(`🛡️ Stop: $${stopLossPrice.toFixed(2)} (${(stopLossDistance * 100).toFixed(2)}%), Balance: $${this.currentBalance.toFixed(2)}`);
    }
    /**
     * Update active positions
     */
    async updateActivePositions(currentData) {
        const positionsToClose = [];
        for (const [positionId, position] of this.activePositions) {
            // Check stop loss
            const hitStopLoss = (position.side === 'LONG' && currentData.price <= position.stopLoss) ||
                (position.side === 'SHORT' && currentData.price >= position.stopLoss);
            if (hitStopLoss) {
                this.closePosition(position, currentData.price, currentData.timestamp, 'Stop Loss');
                positionsToClose.push(positionId);
                continue;
            }
            // Check take profit levels
            for (const tpLevel of position.takeProfitLevels) {
                const hitTakeProfit = (position.side === 'LONG' && currentData.price >= tpLevel) ||
                    (position.side === 'SHORT' && currentData.price <= tpLevel);
                if (hitTakeProfit) {
                    this.closePosition(position, tpLevel, currentData.timestamp, 'Take Profit');
                    positionsToClose.push(positionId);
                    break;
                }
            }
            // Check maximum hold time (24 hours for high leverage)
            const holdTime = currentData.timestamp - position.entryTime;
            if (holdTime > 24 * 60 * 60 * 1000) { // 24 hours
                this.closePosition(position, currentData.price, currentData.timestamp, 'Max Hold Time');
                positionsToClose.push(positionId);
            }
        }
        // Remove closed positions
        positionsToClose.forEach(id => this.activePositions.delete(id));
    }
    /**
     * Close position and calculate P&L
     */
    closePosition(position, exitPrice, exitTime, reason) {
        position.exitPrice = exitPrice;
        position.exitTime = exitTime;
        position.exitReason = reason;
        // Calculate P&L with leverage
        const priceChange = position.side === 'LONG'
            ? (exitPrice - position.entryPrice) / position.entryPrice
            : (position.entryPrice - exitPrice) / position.entryPrice;
        position.pnlPercent = priceChange * 100;
        position.pnl = (priceChange * position.size * position.entryPrice);
        // Update balance
        this.currentBalance += position.pnl;
        // Update peak and drawdown
        if (this.currentBalance > this.peakBalance) {
            this.peakBalance = this.currentBalance;
            this.currentDrawdown = 0;
        }
        else {
            this.currentDrawdown = ((this.peakBalance - this.currentBalance) / this.peakBalance) * 100;
            this.maxDrawdown = Math.max(this.maxDrawdown, this.currentDrawdown);
        }
        this.positions.push(position);
        const holdTimeHours = ((exitTime - position.entryTime) / (1000 * 60 * 60)).toFixed(1);
        logger_1.logger.debug(`💰 Closed ${position.side}: P&L $${position.pnl.toFixed(2)} (${position.pnlPercent.toFixed(2)}%) - ${reason} - Hold: ${holdTimeHours}h`);
    }
    /**
     * Generate comprehensive backtest results
     */
    generateBacktestResults() {
        const winningTrades = this.positions.filter(p => p.pnl > 0);
        const losingTrades = this.positions.filter(p => p.pnl <= 0);
        const totalReturn = this.currentBalance - this.config.initialCapital;
        const totalReturnPercent = (totalReturn / this.config.initialCapital) * 100;
        const returns = this.positions.map(p => p.pnlPercent / 100);
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length || 0;
        const returnStd = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length || 0);
        const sharpeRatio = returnStd > 0 ? (avgReturn / returnStd) * Math.sqrt(252) : 0;
        const grossWins = winningTrades.reduce((sum, p) => sum + p.pnl, 0);
        const grossLosses = Math.abs(losingTrades.reduce((sum, p) => sum + p.pnl, 0));
        const profitFactor = grossLosses > 0 ? grossWins / grossLosses : grossWins > 0 ? 999 : 0;
        return {
            config: this.config,
            summary: {
                totalTrades: this.positions.length,
                winningTrades: winningTrades.length,
                losingTrades: losingTrades.length,
                winRate: this.positions.length > 0 ? (winningTrades.length / this.positions.length) * 100 : 0,
                totalReturn,
                totalReturnPercent,
                maxDrawdown: this.maxDrawdown,
                maxDrawdownPercent: this.maxDrawdown,
                sharpeRatio,
                profitFactor,
                averageWin: winningTrades.length > 0 ? winningTrades.reduce((sum, p) => sum + p.pnl, 0) / winningTrades.length : 0,
                averageLoss: losingTrades.length > 0 ? losingTrades.reduce((sum, p) => sum + p.pnl, 0) / losingTrades.length : 0,
                largestWin: winningTrades.length > 0 ? Math.max(...winningTrades.map(p => p.pnl)) : 0,
                largestLoss: losingTrades.length > 0 ? Math.min(...losingTrades.map(p => p.pnl)) : 0,
                averageHoldTime: this.positions.length > 0 ? this.positions.reduce((sum, p) => sum + (p.exitTime - p.entryTime), 0) / this.positions.length / (1000 * 60 * 60) : 0,
                finalBalance: this.currentBalance
            },
            trades: this.positions,
            dailyReturns: this.calculateDailyReturns(),
            equityCurve: this.dailyBalances,
            monthlyBreakdown: this.calculateMonthlyBreakdown(),
            regimePerformance: this.calculateRegimePerformance()
        };
    }
    // Helper methods for simulations and calculations
    simulateMultiTimeframeAnalysis(data) {
        if (data.length < 20)
            return { signals: { entry: 'HOLD', confidence: 0.3 }, overallTrend: { alignment: 0.5 } };
        const recent = data.slice(-20);
        const prices = recent.map(d => d.close);
        const sma20 = prices.reduce((sum, p) => sum + p, 0) / prices.length;
        const currentPrice = prices[prices.length - 1];
        const trend = currentPrice > sma20 ? 'bullish' : 'bearish';
        const strength = Math.abs(currentPrice - sma20) / sma20;
        const confidence = Math.min(0.9, strength * 10);
        return {
            signals: {
                entry: trend === 'bullish' ? 'BUY' : 'SELL',
                confidence: confidence
            },
            overallTrend: {
                direction: trend,
                strength: Math.min(0.9, strength * 5),
                alignment: confidence
            }
        };
    }
    simulateRegimeDetection(data) {
        if (data.length < 10)
            return { current_regime: 'sideways', confidence: 0.5 };
        const recent = data.slice(-10);
        const volatility = this.calculateVolatility(recent.map(d => d.close));
        let regime = 'sideways';
        if (volatility > 0.05)
            regime = 'volatile';
        else if (volatility < 0.02)
            regime = 'consolidation';
        else
            regime = 'trending_bullish';
        return {
            current_regime: regime,
            confidence: 0.7 + Math.random() * 0.2
        };
    }
    simulateMLPrediction(data, mtfAnalysis, regimeAnalysis) {
        const confidence = (mtfAnalysis.signals.confidence + regimeAnalysis.confidence) / 2;
        return {
            ensemble_confidence: confidence,
            recommendation: mtfAnalysis.signals.entry === 'BUY' ? 'STRONG_BUY' : 'STRONG_SELL'
        };
    }
    evaluateIntelligentOpportunity(currentData, mtfAnalysis, regimeAnalysis, mlPrediction) {
        let score = 50;
        // Signal strength
        score += mtfAnalysis.signals.confidence * 30;
        // Trend alignment
        score += mtfAnalysis.overallTrend.alignment * 20;
        // ML confidence
        score += mlPrediction.ensemble_confidence * 20;
        // Regime bonus
        if (regimeAnalysis.current_regime.includes('trending'))
            score += 10;
        const side = mtfAnalysis.signals.entry === 'BUY' ? 'LONG' : 'SHORT';
        return {
            shouldTrade: score >= 60,
            score,
            side,
            regime: regimeAnalysis.current_regime,
            signals: mtfAnalysis.signals
        };
    }
    calculateVolatility(prices) {
        if (prices.length < 2)
            return 0;
        const returns = prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
        return Math.sqrt(variance);
    }
    updateDailyTracking(date) {
        this.dailyBalances.push({
            date,
            balance: this.currentBalance,
            drawdown: this.currentDrawdown
        });
    }
    calculateDailyReturns() {
        return this.dailyBalances.slice(1).map((day, i) => (day.balance - this.dailyBalances[i].balance) / this.dailyBalances[i].balance);
    }
    calculateMonthlyBreakdown() {
        const monthly = {};
        this.positions.forEach(trade => {
            const month = new Date(trade.entryTime).toISOString().substr(0, 7);
            if (!monthly[month])
                monthly[month] = { trades: 0, pnl: 0, wins: 0 };
            monthly[month].trades++;
            monthly[month].pnl += trade.pnl;
            if (trade.pnl > 0)
                monthly[month].wins++;
        });
        return Object.entries(monthly).map(([month, data]) => ({
            month,
            trades: data.trades,
            pnl: data.pnl,
            winRate: data.trades > 0 ? (data.wins / data.trades) * 100 : 0
        }));
    }
    calculateRegimePerformance() {
        const regimes = {};
        this.positions.forEach(trade => {
            const regime = trade.regimeAtEntry;
            if (!regimes[regime])
                regimes[regime] = { trades: 0, wins: 0, totalReturn: 0 };
            regimes[regime].trades++;
            regimes[regime].totalReturn += trade.pnlPercent;
            if (trade.pnl > 0)
                regimes[regime].wins++;
        });
        return Object.entries(regimes).map(([regime, data]) => ({
            regime,
            trades: data.trades,
            winRate: data.trades > 0 ? (data.wins / data.trades) * 100 : 0,
            avgReturn: data.trades > 0 ? data.totalReturn / data.trades : 0
        }));
    }
    closeAllPositions(finalData) {
        for (const [positionId, position] of this.activePositions) {
            this.closePosition(position, finalData.price, finalData.timestamp, 'Backtest End');
        }
        this.activePositions.clear();
    }
    createMockDataService() {
        return {
            getMultiTimeframeData: () => Promise.resolve({}),
            initialize: () => Promise.resolve()
        };
    }
    logBacktestSummary(results) {
        logger_1.logger.info('\n🎯 INTELLIGENT TRADING BOT BACKTEST RESULTS:');
        logger_1.logger.info('═══════════════════════════════════════════════');
        logger_1.logger.info(`📊 Total Trades: ${results.summary.totalTrades}`);
        logger_1.logger.info(`✅ Win Rate: ${results.summary.winRate.toFixed(1)}%`);
        logger_1.logger.info(`💰 Total Return: $${results.summary.totalReturn.toFixed(2)} (${results.summary.totalReturnPercent.toFixed(1)}%)`);
        logger_1.logger.info(`📈 Final Balance: $${results.summary.finalBalance.toFixed(2)}`);
        logger_1.logger.info(`📉 Max Drawdown: ${results.summary.maxDrawdownPercent.toFixed(1)}%`);
        logger_1.logger.info(`⚡ Sharpe Ratio: ${results.summary.sharpeRatio.toFixed(2)}`);
        logger_1.logger.info(`🎯 Profit Factor: ${results.summary.profitFactor.toFixed(2)}`);
        logger_1.logger.info(`⏱️ Avg Hold Time: ${results.summary.averageHoldTime.toFixed(1)} hours`);
        logger_1.logger.info(`🏆 Largest Win: $${results.summary.largestWin.toFixed(2)}`);
        logger_1.logger.info(`💥 Largest Loss: $${results.summary.largestLoss.toFixed(2)}`);
        if (results.summary.totalReturnPercent > 0) {
            logger_1.logger.info(`🚀 PROFITABLE STRATEGY! ${results.summary.totalReturnPercent.toFixed(1)}% return with dynamic leverage`);
        }
        else {
            logger_1.logger.info(`⚠️ Strategy needs optimization. Loss: ${results.summary.totalReturnPercent.toFixed(1)}%`);
        }
    }
    /**
     * Calculate dynamic risk based on current balance
     * Risk Ladder Strategy: Start ultra-aggressive, become conservative as balance grows
     */
    calculateDynamicRisk() {
        const balanceMultiplier = this.currentBalance / this.config.initialCapital;
        // Phase 1: Survival Mode ($10-$50) - ULTRA AGGRESSIVE
        if (balanceMultiplier <= 5) {
            return {
                riskPercent: 40, // 40% risk per trade
                stopLossPercent: 0.3, // 0.3% stop loss (very tight)
                takeProfitLevels: [0.008, 0.015, 0.025], // 0.8%, 1.5%, 2.5% take profits
                phase: 'SURVIVAL_MODE'
            };
        }
        // Phase 2: Growth Mode ($50-$200) - AGGRESSIVE
        else if (balanceMultiplier <= 20) {
            return {
                riskPercent: 25, // 25% risk per trade
                stopLossPercent: 0.5, // 0.5% stop loss
                takeProfitLevels: [0.01, 0.02, 0.035], // 1%, 2%, 3.5% take profits
                phase: 'GROWTH_MODE'
            };
        }
        // Phase 3: Expansion Mode ($200-$1000) - MODERATE
        else if (balanceMultiplier <= 100) {
            return {
                riskPercent: 15, // 15% risk per trade
                stopLossPercent: 0.8, // 0.8% stop loss
                takeProfitLevels: [0.012, 0.025, 0.04], // 1.2%, 2.5%, 4% take profits
                phase: 'EXPANSION_MODE'
            };
        }
        // Phase 4: Consolidation Mode ($1000-$5000) - CONSERVATIVE
        else if (balanceMultiplier <= 500) {
            return {
                riskPercent: 8, // 8% risk per trade
                stopLossPercent: 1.2, // 1.2% stop loss
                takeProfitLevels: [0.015, 0.03, 0.05], // 1.5%, 3%, 5% take profits
                phase: 'CONSOLIDATION_MODE'
            };
        }
        // Phase 5: Wealth Preservation Mode ($5000+) - ULTRA CONSERVATIVE
        else {
            return {
                riskPercent: 3, // 3% risk per trade
                stopLossPercent: 2.0, // 2% stop loss
                takeProfitLevels: [0.02, 0.04, 0.06], // 2%, 4%, 6% take profits
                phase: 'WEALTH_PRESERVATION'
            };
        }
    }
    /**
     * Calculate dynamic leverage based on current balance
     * Leverage Ladder: Start extreme, reduce as balance grows
     */
    calculateDynamicLeverage() {
        const balanceMultiplier = this.currentBalance / this.config.initialCapital;
        // Phase 1: Survival Mode ($10-$50) - MAXIMUM LEVERAGE
        if (balanceMultiplier <= 5) {
            return {
                leverage: 200, // 200x leverage (EXTREME!)
                phase: 'MAXIMUM_LEVERAGE'
            };
        }
        // Phase 2: Growth Mode ($50-$200) - HIGH LEVERAGE
        else if (balanceMultiplier <= 20) {
            return {
                leverage: 100, // 100x leverage
                phase: 'HIGH_LEVERAGE'
            };
        }
        // Phase 3: Expansion Mode ($200-$1000) - MODERATE LEVERAGE
        else if (balanceMultiplier <= 100) {
            return {
                leverage: 50, // 50x leverage
                phase: 'MODERATE_LEVERAGE'
            };
        }
        // Phase 4: Consolidation Mode ($1000-$5000) - LOW LEVERAGE
        else if (balanceMultiplier <= 500) {
            return {
                leverage: 20, // 20x leverage
                phase: 'LOW_LEVERAGE'
            };
        }
        // Phase 5: Wealth Preservation Mode ($5000+) - MINIMAL LEVERAGE
        else {
            return {
                leverage: 10, // 10x leverage
                phase: 'MINIMAL_LEVERAGE'
            };
        }
    }
}
exports.IntelligentTradingBotBacktester = IntelligentTradingBotBacktester;
//# sourceMappingURL=IntelligentTradingBotBacktester.js.map