#!/usr/bin/env node
"use strict";
/**
 * Simulate Position Management
 * Demonstrates AI position management with simulated Delta position
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PositionSimulator = void 0;
const aiPositionManager_1 = require("../services/aiPositionManager");
const logger_1 = require("../utils/logger");
class PositionSimulator {
    constructor() {
        this.priceDirection = 1;
        this.iteration = 0;
        // Create a mock Delta API for simulation
        const mockDeltaApi = this.createMockDeltaApi();
        this.aiManager = new aiPositionManager_1.AIPositionManager(mockDeltaApi);
        // Create a simulated position (like the one you might have)
        this.simulatedPosition = {
            symbol: 'BTC_USDT',
            size: '0.01', // 0.01 BTC long position
            entry_price: '95000', // Entry at $95,000
            unrealized_pnl: '0',
        };
        this.currentPrice = 95000; // Starting price
    }
    /**
     * Create mock Delta API for simulation
     */
    createMockDeltaApi() {
        return {
            getPositions: async () => [this.simulatedPosition],
            getTicker: async (symbol) => ({
                close: this.currentPrice.toString(),
                last_price: this.currentPrice.toString(),
            }),
            // Mock other methods as needed
        };
    }
    /**
     * Run position management simulation
     */
    async runSimulation() {
        logger_1.logger.info('🎮 STARTING POSITION MANAGEMENT SIMULATION');
        logger_1.logger.info('='.repeat(80));
        logger_1.logger.info('📊 SIMULATED POSITION:');
        logger_1.logger.info(`   Symbol: ${this.simulatedPosition.symbol}`);
        logger_1.logger.info(`   Size: ${this.simulatedPosition.size} BTC (LONG)`);
        logger_1.logger.info(`   Entry Price: $${this.simulatedPosition.entry_price}`);
        logger_1.logger.info(`   Current Price: $${this.currentPrice.toFixed(2)}`);
        logger_1.logger.info('\n🤖 Starting AI Position Management...');
        // Start AI management
        await this.aiManager.startManagement();
        // Simulate price movements and AI responses
        const maxIterations = 20;
        for (let i = 0; i < maxIterations; i++) {
            this.iteration = i + 1;
            // Simulate price movement
            this.simulateMarketMovement();
            // Update position P&L
            this.updatePositionPnL();
            // Log current state
            this.logCurrentState();
            // Wait 3 seconds between iterations
            await this.sleep(3000);
            // Check if position was closed by AI
            const managedPositions = this.aiManager.getManagedPositions();
            const activePosition = managedPositions.find(p => p.status === 'MANAGING');
            if (!activePosition) {
                logger_1.logger.info('🏁 Position closed by AI - simulation complete');
                break;
            }
        }
        // Stop AI management
        this.aiManager.stopManagement();
        // Display final results
        this.displayFinalResults();
    }
    /**
     * Simulate realistic market movement
     */
    simulateMarketMovement() {
        // Simulate realistic BTC price movement
        const volatility = 0.02; // 2% volatility
        const randomChange = (Math.random() - 0.5) * 2 * volatility;
        // Add some trend (60% chance to continue current direction)
        if (Math.random() > 0.4) {
            this.priceDirection *= -1; // Change direction
        }
        const trendInfluence = this.priceDirection * 0.005; // 0.5% trend
        const totalChange = randomChange + trendInfluence;
        this.currentPrice *= (1 + totalChange);
        // Keep price within reasonable bounds
        this.currentPrice = Math.max(80000, Math.min(120000, this.currentPrice));
    }
    /**
     * Update position P&L
     */
    updatePositionPnL() {
        const entryPrice = parseFloat(this.simulatedPosition.entry_price);
        const size = parseFloat(this.simulatedPosition.size);
        const priceChange = this.currentPrice - entryPrice;
        const unrealizedPnl = priceChange * size;
        this.simulatedPosition.unrealized_pnl = unrealizedPnl.toString();
    }
    /**
     * Log current state
     */
    logCurrentState() {
        const entryPrice = parseFloat(this.simulatedPosition.entry_price);
        const unrealizedPnl = parseFloat(this.simulatedPosition.unrealized_pnl);
        const profitPercent = ((this.currentPrice - entryPrice) / entryPrice) * 100;
        logger_1.logger.info(`\n📊 Iteration ${this.iteration}:`);
        logger_1.logger.info(`   Current Price: $${this.currentPrice.toFixed(2)}`);
        logger_1.logger.info(`   Unrealized P&L: $${unrealizedPnl.toFixed(2)} (${profitPercent.toFixed(2)}%)`);
        // Show AI management status
        const managedPositions = this.aiManager.getManagedPositions();
        if (managedPositions.length > 0) {
            const position = managedPositions[0];
            const executedExits = position.partialExits.filter(e => e.executed).length;
            logger_1.logger.info(`   AI Status: ${position.status}`);
            logger_1.logger.info(`   Partial Exits: ${executedExits}/${position.partialExits.length}`);
            logger_1.logger.info(`   Stop Loss: $${position.stopLoss.toFixed(2)}`);
            if (position.aiRecommendations.length > 0) {
                const lastRecommendation = position.aiRecommendations[position.aiRecommendations.length - 1];
                logger_1.logger.info(`   Last AI Action: ${lastRecommendation.split(': ')[1]}`);
            }
        }
    }
    /**
     * Display final results
     */
    displayFinalResults() {
        const managedPositions = this.aiManager.getManagedPositions();
        logger_1.logger.info('\n🎯 SIMULATION RESULTS:');
        logger_1.logger.info('='.repeat(80));
        if (managedPositions.length > 0) {
            const position = managedPositions[0];
            const entryPrice = parseFloat(this.simulatedPosition.entry_price);
            const finalPnl = parseFloat(this.simulatedPosition.unrealized_pnl);
            const profitPercent = ((this.currentPrice - entryPrice) / entryPrice) * 100;
            const executedExits = position.partialExits.filter(e => e.executed);
            logger_1.logger.info('📊 POSITION SUMMARY:');
            logger_1.logger.info(`   Entry Price: $${entryPrice.toFixed(2)}`);
            logger_1.logger.info(`   Final Price: $${this.currentPrice.toFixed(2)}`);
            logger_1.logger.info(`   Total P&L: $${finalPnl.toFixed(2)} (${profitPercent.toFixed(2)}%)`);
            logger_1.logger.info(`   Position Status: ${position.status}`);
            logger_1.logger.info('\n🤖 AI MANAGEMENT PERFORMANCE:');
            logger_1.logger.info(`   Partial Exits Executed: ${executedExits.length}/${position.partialExits.length}`);
            if (executedExits.length > 0) {
                logger_1.logger.info('   Exit Details:');
                executedExits.forEach(exit => {
                    logger_1.logger.info(`     Level ${exit.level}: ${exit.percentage}% at $${exit.targetPrice.toFixed(2)}`);
                });
            }
            logger_1.logger.info(`   Final Stop Loss: $${position.stopLoss.toFixed(2)}`);
            logger_1.logger.info(`   Total AI Recommendations: ${position.aiRecommendations.length}`);
            if (position.aiRecommendations.length > 0) {
                logger_1.logger.info('\n📝 AI RECOMMENDATION HISTORY:');
                position.aiRecommendations.forEach((rec, index) => {
                    const [timestamp, action] = rec.split(': ');
                    logger_1.logger.info(`   ${index + 1}. ${action}`);
                });
            }
        }
        logger_1.logger.info('\n✅ SIMULATION COMPLETE');
        logger_1.logger.info('🚀 This demonstrates how the AI will manage your real Delta Exchange position!');
    }
    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.PositionSimulator = PositionSimulator;
/**
 * Main execution
 */
async function main() {
    const simulator = new PositionSimulator();
    await simulator.runSimulation();
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        logger_1.logger.error('💥 Simulation failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=simulate-position-management.js.map