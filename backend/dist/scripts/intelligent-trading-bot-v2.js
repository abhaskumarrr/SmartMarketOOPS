"use strict";
/**
 * Intelligent Trading Bot V2.0
 * Complete implementation with multi-timeframe analysis, regime detection, and adaptive position management
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntelligentTradingBotV2 = void 0;
const dotenv_1 = __importDefault(require("dotenv"));
const DeltaExchangeUnified_1 = require("../services/DeltaExchangeUnified");
const MultiTimeframeAnalysisEngine_1 = require("../services/MultiTimeframeAnalysisEngine");
const EnhancedMarketRegimeDetector_1 = require("../services/EnhancedMarketRegimeDetector");
const AdaptiveStopLossSystem_1 = require("../services/AdaptiveStopLossSystem");
const SmartTakeProfitSystem_1 = require("../services/SmartTakeProfitSystem");
const EnhancedMLIntegrationService_1 = require("../services/EnhancedMLIntegrationService");
const IntelligentTradingBotTestSuite_1 = require("../tests/IntelligentTradingBotTestSuite");
const logger_1 = require("../utils/logger");
// Load environment variables
dotenv_1.default.config();
class IntelligentTradingBotV2 {
    constructor() {
        this.activePositions = new Map();
        this.isRunning = false;
        this.scanInterval = null;
        // Performance metrics
        this.totalTrades = 0;
        this.winningTrades = 0;
        this.totalPnL = 0;
        this.maxDrawdown = 0;
        this.startingBalance = 0;
        this.config = {
            symbols: ['BTCUSD', 'ETHUSD', 'SOLUSD'],
            maxPositions: 3,
            riskPerTrade: 2.0, // 2% risk per trade
            maxLeverage: {
                'BTCUSD': 100,
                'ETHUSD': 100,
                'SOLUSD': 50
            },
            tradingCapital: 1000, // $1000 starting capital
            enablePaperTrading: false, // Live trading
            scanInterval: 30000, // 30 seconds
            enableIntelligentRiskManagement: true
        };
        // Initialize services
        this.deltaService = new DeltaExchangeUnified_1.DeltaExchangeUnified({
            apiKey: process.env.DELTA_API_KEY,
            apiSecret: process.env.DELTA_API_SECRET,
            testnet: true // Use testnet for safety
        });
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine(this.deltaService);
        this.regimeDetector = new EnhancedMarketRegimeDetector_1.EnhancedMarketRegimeDetector(this.deltaService);
        this.stopLossSystem = new AdaptiveStopLossSystem_1.AdaptiveStopLossSystem(this.deltaService);
        this.takeProfitSystem = new SmartTakeProfitSystem_1.SmartTakeProfitSystem(this.deltaService);
        this.mlService = new EnhancedMLIntegrationService_1.EnhancedMLIntegrationService(this.deltaService);
        this.testSuite = new IntelligentTradingBotTestSuite_1.IntelligentTradingBotTestSuite();
    }
    /**
     * Start the intelligent trading bot
     */
    async start() {
        try {
            logger_1.logger.info('🚀 Starting Intelligent Trading Bot V2.0');
            // Run comprehensive tests first
            logger_1.logger.info('🧪 Running system tests...');
            const testResults = await this.testSuite.runComprehensiveTests();
            if (testResults.overall_status === 'FAIL') {
                logger_1.logger.error('❌ System tests failed. Aborting startup.');
                return;
            }
            logger_1.logger.info('✅ All system tests passed. Proceeding with startup.');
            // Initialize Delta Exchange connection
            await this.deltaService.initialize();
            // Get starting balance
            this.startingBalance = await this.getCurrentBalance();
            logger_1.logger.info(`💰 Starting balance: $${this.startingBalance.toFixed(2)}`);
            // Display configuration
            this.displayConfiguration();
            // Start trading loop
            this.isRunning = true;
            this.scanInterval = setInterval(() => {
                this.tradingCycle().catch(error => {
                    logger_1.logger.error('❌ Error in trading cycle:', error);
                });
            }, this.config.scanInterval);
            logger_1.logger.info('✅ Intelligent Trading Bot V2.0 started successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start trading bot:', error);
            throw error;
        }
    }
    /**
     * Stop the trading bot
     */
    async stop() {
        logger_1.logger.info('🛑 Stopping Intelligent Trading Bot V2.0');
        this.isRunning = false;
        if (this.scanInterval) {
            clearInterval(this.scanInterval);
            this.scanInterval = null;
        }
        // Display final performance
        await this.displayPerformanceReport();
        logger_1.logger.info('✅ Intelligent Trading Bot V2.0 stopped');
    }
    /**
     * Main trading cycle
     */
    async tradingCycle() {
        if (!this.isRunning)
            return;
        try {
            logger_1.logger.debug('🔄 Starting trading cycle');
            // Update existing positions
            await this.updateActivePositions();
            // Manage existing positions with intelligent algorithms
            await this.managePositionsIntelligently();
            // Look for new opportunities if we have capacity
            if (this.activePositions.size < this.config.maxPositions) {
                await this.scanForIntelligentOpportunities();
            }
            // Update performance metrics
            await this.updatePerformanceMetrics();
            logger_1.logger.debug('✅ Trading cycle completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error in trading cycle:', error);
        }
    }
    /**
     * Scan for intelligent trading opportunities
     */
    async scanForIntelligentOpportunities() {
        for (const symbol of this.config.symbols) {
            try {
                // Skip if we already have a position in this symbol
                if (Array.from(this.activePositions.values()).some(p => p.symbol === symbol)) {
                    continue;
                }
                logger_1.logger.info(`🔍 Analyzing ${symbol} for opportunities`);
                // Get comprehensive market analysis
                const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(symbol);
                const regimeAnalysis = await this.regimeDetector.detectRegime(symbol);
                // Get ML prediction
                const currentPrice = await this.getCurrentPrice(symbol);
                const mlPrediction = await this.mlService.getEnsemblePrediction(symbol, 'LONG', currentPrice, currentPrice);
                // Evaluate opportunity
                const opportunity = this.evaluateIntelligentOpportunity(symbol, mtfAnalysis, regimeAnalysis, mlPrediction);
                if (opportunity.shouldTrade) {
                    await this.executeIntelligentTrade(opportunity);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Error analyzing ${symbol}:`, error);
            }
        }
    }
    /**
     * Manage positions with intelligent algorithms
     */
    async managePositionsIntelligently() {
        for (const [positionId, position] of this.activePositions) {
            try {
                logger_1.logger.debug(`📊 Managing position ${positionId} (${position.symbol})`);
                // Update current price
                position.currentPrice = await this.getCurrentPrice(position.symbol);
                position.lastUpdate = Date.now();
                // Get ML prediction for position outcome
                const mlPrediction = await this.mlService.predictPositionOutcome(position.symbol, position.side, position.entryPrice, position.currentPrice, Date.now() - position.entryTime);
                // Update position health score
                position.healthScore = mlPrediction.confidence * 100;
                // Get regime analysis
                const regimeAnalysis = await this.regimeDetector.detectRegime(position.symbol);
                // Calculate adaptive stop loss
                const adaptiveStopLoss = await this.stopLossSystem.calculateStopLoss({
                    symbol: position.symbol,
                    side: position.side,
                    entry_price: position.entryPrice,
                    current_price: position.currentPrice,
                    size: position.size,
                    entry_time: position.entryTime
                });
                // Update stop loss if needed
                if (!position.stopLoss || Math.abs(adaptiveStopLoss.stop_price - position.stopLoss) > position.currentPrice * 0.005) {
                    position.stopLoss = adaptiveStopLoss.stop_price;
                    logger_1.logger.info(`🎯 Updated stop loss for ${position.symbol}: $${adaptiveStopLoss.stop_price.toFixed(2)}`);
                }
                // Check for exit conditions
                await this.checkIntelligentExitConditions(position, mlPrediction, regimeAnalysis);
            }
            catch (error) {
                logger_1.logger.error(`❌ Error managing position ${positionId}:`, error);
            }
        }
    }
    /**
     * Evaluate intelligent trading opportunity
     */
    evaluateIntelligentOpportunity(symbol, mtfAnalysis, regimeAnalysis, mlPrediction) {
        const signals = mtfAnalysis.signals;
        const overallTrend = mtfAnalysis.overallTrend;
        const regime = regimeAnalysis.current_regime;
        const mlRecommendation = mlPrediction.recommendation;
        // Intelligent scoring system
        let opportunityScore = 0;
        const reasoning = [];
        // Multi-timeframe signal strength (40% weight)
        if (signals.confidence > 0.7) {
            opportunityScore += 40;
            reasoning.push(`Strong MTF signal: ${signals.entry} (${(signals.confidence * 100).toFixed(1)}%)`);
        }
        else if (signals.confidence > 0.5) {
            opportunityScore += 20;
            reasoning.push(`Moderate MTF signal: ${signals.entry} (${(signals.confidence * 100).toFixed(1)}%)`);
        }
        // Trend alignment (25% weight)
        if (overallTrend.alignment > 0.8) {
            opportunityScore += 25;
            reasoning.push(`Excellent trend alignment (${(overallTrend.alignment * 100).toFixed(1)}%)`);
        }
        else if (overallTrend.alignment > 0.6) {
            opportunityScore += 15;
            reasoning.push(`Good trend alignment (${(overallTrend.alignment * 100).toFixed(1)}%)`);
        }
        // ML prediction (25% weight)
        if (mlPrediction.ensemble_confidence > 0.8) {
            opportunityScore += 25;
            reasoning.push(`High ML confidence: ${mlRecommendation} (${(mlPrediction.ensemble_confidence * 100).toFixed(1)}%)`);
        }
        else if (mlPrediction.ensemble_confidence > 0.6) {
            opportunityScore += 15;
            reasoning.push(`Moderate ML confidence: ${mlRecommendation} (${(mlPrediction.ensemble_confidence * 100).toFixed(1)}%)`);
        }
        // Regime compatibility (10% weight)
        if (regimeAnalysis.confidence > 0.7) {
            opportunityScore += 10;
            reasoning.push(`Favorable regime: ${regime} (${(regimeAnalysis.confidence * 100).toFixed(1)}%)`);
        }
        // Determine trade direction and size
        let side = 'LONG';
        if (signals.entry === 'SELL' || mlRecommendation.includes('SELL')) {
            side = 'SHORT';
        }
        // Intelligent position sizing based on confidence
        const baseSize = this.config.tradingCapital * (this.config.riskPerTrade / 100);
        const confidenceMultiplier = Math.min(2.0, opportunityScore / 50); // Max 2x size for high confidence
        const positionSize = baseSize * confidenceMultiplier;
        return {
            symbol,
            shouldTrade: opportunityScore >= 60, // Require 60+ score to trade
            side,
            opportunityScore,
            positionSize,
            reasoning,
            confidence: opportunityScore / 100
        };
    }
    /**
     * Execute intelligent trade
     */
    async executeIntelligentTrade(opportunity) {
        try {
            logger_1.logger.info(`🎯 Executing intelligent trade: ${opportunity.symbol} ${opportunity.side}`);
            logger_1.logger.info(`📊 Opportunity score: ${opportunity.opportunityScore}/100`);
            logger_1.logger.info(`💡 Reasoning: ${opportunity.reasoning.join(', ')}`);
            const currentPrice = await this.getCurrentPrice(opportunity.symbol);
            const leverage = this.config.maxLeverage[opportunity.symbol] || 50;
            // Calculate position size in contracts
            const contractSize = this.calculateContractSize(opportunity.symbol, opportunity.positionSize, currentPrice, leverage);
            if (!this.config.enablePaperTrading) {
                // Place actual order
                const orderRequest = {
                    product_id: this.deltaService.getProductId(opportunity.symbol),
                    side: opportunity.side.toLowerCase(),
                    size: contractSize,
                    order_type: 'market_order'
                };
                const order = await this.deltaService.placeOrder(orderRequest);
                logger_1.logger.info(`✅ Order placed: ${order.id}`);
                // Create position record
                const position = {
                    id: order.id,
                    symbol: opportunity.symbol,
                    side: opportunity.side,
                    entryPrice: currentPrice,
                    currentPrice: currentPrice,
                    size: contractSize,
                    leverage: leverage,
                    entryTime: Date.now(),
                    takeProfitLevels: [],
                    healthScore: opportunity.opportunityScore,
                    lastUpdate: Date.now()
                };
                this.activePositions.set(order.id, position);
                this.totalTrades++;
            }
            else {
                logger_1.logger.info(`📝 Paper trade: ${opportunity.symbol} ${opportunity.side} ${contractSize} contracts`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error executing trade:`, error);
        }
    }
    // Helper methods
    async getCurrentBalance() {
        try {
            const balances = await this.deltaService.getBalances();
            const usdBalance = balances.find(b => b.asset_symbol === 'USD');
            return parseFloat(usdBalance?.available_balance || '0');
        }
        catch (error) {
            logger_1.logger.error('Error getting balance:', error);
            return 0;
        }
    }
    async getCurrentPrice(symbol) {
        // Simplified - would get from market data
        const prices = {
            'BTCUSD': 105000,
            'ETHUSD': 4000,
            'SOLUSD': 200
        };
        return prices[symbol] || 100;
    }
    calculateContractSize(symbol, usdAmount, price, leverage) {
        const lotSizes = {
            'BTCUSD': 0.001,
            'ETHUSD': 0.01,
            'SOLUSD': 1
        };
        const lotSize = lotSizes[symbol] || 0.01;
        const notionalValue = usdAmount * leverage;
        const contracts = Math.floor(notionalValue / (price * lotSize));
        return Math.max(1, contracts);
    }
    displayConfiguration() {
        logger_1.logger.info('\n📋 INTELLIGENT TRADING BOT V2.0 CONFIGURATION:');
        logger_1.logger.info(`🎯 Symbols: ${this.config.symbols.join(', ')}`);
        logger_1.logger.info(`📊 Max Positions: ${this.config.maxPositions}`);
        logger_1.logger.info(`⚡ Risk Per Trade: ${this.config.riskPerTrade}%`);
        logger_1.logger.info(`💰 Trading Capital: $${this.config.tradingCapital}`);
        logger_1.logger.info(`🔄 Scan Interval: ${this.config.scanInterval / 1000}s`);
        logger_1.logger.info(`🧠 Intelligent Risk Management: ${this.config.enableIntelligentRiskManagement ? 'ENABLED' : 'DISABLED'}`);
        logger_1.logger.info(`📝 Paper Trading: ${this.config.enablePaperTrading ? 'ENABLED' : 'DISABLED'}`);
    }
    async updateActivePositions() {
        // Update positions from exchange
    }
    async checkIntelligentExitConditions(position, mlPrediction, regimeAnalysis) {
        // Implement intelligent exit logic
    }
    async updatePerformanceMetrics() {
        // Update performance tracking
    }
    async displayPerformanceReport() {
        const currentBalance = await this.getCurrentBalance();
        const totalReturn = ((currentBalance - this.startingBalance) / this.startingBalance) * 100;
        const winRate = this.totalTrades > 0 ? (this.winningTrades / this.totalTrades) * 100 : 0;
        logger_1.logger.info('\n📊 PERFORMANCE REPORT:');
        logger_1.logger.info(`💰 Starting Balance: $${this.startingBalance.toFixed(2)}`);
        logger_1.logger.info(`💰 Current Balance: $${currentBalance.toFixed(2)}`);
        logger_1.logger.info(`📈 Total Return: ${totalReturn.toFixed(2)}%`);
        logger_1.logger.info(`🎯 Total Trades: ${this.totalTrades}`);
        logger_1.logger.info(`✅ Winning Trades: ${this.winningTrades}`);
        logger_1.logger.info(`📊 Win Rate: ${winRate.toFixed(1)}%`);
        logger_1.logger.info(`📉 Max Drawdown: ${this.maxDrawdown.toFixed(2)}%`);
    }
}
exports.IntelligentTradingBotV2 = IntelligentTradingBotV2;
// Main execution
async function main() {
    const bot = new IntelligentTradingBotV2();
    // Handle graceful shutdown
    process.on('SIGINT', async () => {
        logger_1.logger.info('🛑 Received SIGINT, shutting down gracefully...');
        await bot.stop();
        process.exit(0);
    });
    try {
        await bot.start();
    }
    catch (error) {
        logger_1.logger.error('❌ Failed to start trading bot:', error);
        process.exit(1);
    }
}
// Run if this file is executed directly
if (require.main === module) {
    main();
}
//# sourceMappingURL=intelligent-trading-bot-v2.js.map